#!/usr/bin/env python3
"""
Stroke2Font DQN Inference — Apply trained model to any character data
=====================================================================

Usage:
    # 处理默认数据集 (paper-1-180xk.json 全部180字)
    python src/dqn_inference.py

    # 处理指定JSON文件
    python src/dqn_inference.py --input data/hanzi_dataset_150_chars.json

    # 指定输出目录
    python src/dqn_inference.py --input data/my_chars.json --output output/

    # 只处理前N个字
    python src/dqn_inference.py --limit 50

    # 指定优化方法: dqn / ai_adaptive / baseline / all
    python src/dqn_inference.py --method all

    # 生成可视化
    python src/dqn_inference.py --visualize

Input JSON format (与训练数据一致):
    [
        {
            "character": "啊",
            "strokes": [
                {
                    "strokeType": "横向笔画",
                    "coordinates": [[x1,y1], [x2,y2], ...],
                    "pointCount": 12
                },
                ...
            ]
        },
        ...
    ]

Output:
    output/
      optimized_results.json    — 每字每笔的优化坐标 + 指标
      summary.txt               — 汇总统计
      visualize_*.png           — 可视化 (--visualize)
"""

import argparse
import json
import math
import os
import sys
import time
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Stroke types (must match training) ───────────────────────────────────
STROKE_TYPES = ['横向笔画', '纵向笔画', '撇捺笔画', '折转笔画', '复合笔画', '点']


# ============================================================================
# Model
# ============================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim=10, action_dim=20, hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class DQNInference:
    """Load trained DQN model and run inference on new characters."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.delta_alpha_values = [-10, -5, 0, 5, 10]
        self.sigma_values = [0.5, 0.8, 1.0, 1.5]
        self.actions_lut = [
            (da, s)
            for da in self.delta_alpha_values
            for s in self.sigma_values
        ]

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device,
                                weights_only=True)
        cfg = checkpoint['config']
        self.q_net = QNetwork(
            cfg['state_dim'], cfg['action_dim'],
            cfg['hidden1'], cfg['hidden2'])
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.q_net.to(self.device)
        self.q_net.eval()
        print(f"[OK] Model loaded: {model_path}")
        print(f"     Network: {cfg['state_dim']}→{cfg['hidden1']}"
              f"→{cfg['hidden2']}→{cfg['action_dim']}")

    def build_state(self, stroke: dict, complexity: float,
                    n_strokes: int, stroke_idx: int) -> np.ndarray:
        """Build 10-dim state vector for a stroke."""
        stype = stroke.get('strokeType', '点')
        tidx = STROKE_TYPES.index(stype) if stype in STROKE_TYPES else 5
        oh = [0.0] * 6
        oh[tidx] = 1.0

        c_norm = min(complexity / 100.0, 1.0)

        coords = stroke.get('coordinates', [])
        n_pts = len(coords)
        if n_pts >= 2:
            length = sum(
                math.sqrt((coords[i][0] - coords[i-1][0])**2 +
                           (coords[i][1] - coords[i-1][1])**2)
                for i in range(1, n_pts))
            density = min(n_pts / (length + 1e-10), 1.0)
            length_norm = min(length / 100.0, 1.0)
        else:
            density, length_norm = 0.0, 0.0

        budget = 1.0 - stroke_idx / max(n_strokes, 1)
        return np.array(
            oh + [c_norm, density, length_norm, budget], dtype=np.float32)

    def select_action(self, state: np.ndarray) -> int:
        """Greedy action selection."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.q_net(state_t)
            return int(q.argmax(dim=1).item())

    def decode_action(self, action: int):
        """Action index → (Δα degrees, σ)."""
        return self.actions_lut[action]

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return full Q-value vector (for analysis)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(state_t).cpu().numpy()[0]


# ============================================================================
# Pipeline functions
# ============================================================================

def extract_coords(stroke):
    coords = stroke.get('coordinates', [])
    x = np.array([float(p[0]) for p in coords if len(p) >= 2])
    y = np.array([float(p[1]) for p in coords if len(p) >= 2])
    return x, y


def rotate_coords(x, y, delta_deg):
    if len(x) < 2 or abs(delta_deg) < 1e-6:
        return x.copy(), y.copy()
    cx, cy = np.mean(x), np.mean(y)
    rad = np.radians(delta_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx, dy = x - cx, y - cy
    return cx + dx*cos_a - dy*sin_a, cy + dx*sin_a + dy*cos_a


def baseline_interpolation(x, y):
    if len(x) < 2: return x, y
    t = np.linspace(0, 1, len(x))
    n_pts = max(len(x), min(50, len(x)*3))
    t_new = np.linspace(0, 1, n_pts)
    return interp1d(t, x)(t_new), interp1d(t, y)(t_new)


def ai_optimized_interpolation(x, y, sigma=1.0, num_points=50):
    if len(x) < 2: return x, y
    t = np.linspace(0, 1, len(x))
    n_pts = max(len(x), min(num_points, len(x)*3))
    t_new = np.linspace(0, 1, n_pts)
    kind = 'cubic' if len(x) >= 4 else 'linear'
    x_new = interp1d(t, x, kind=kind)(t_new)
    y_new = interp1d(t, y, kind=kind)(t_new)
    if sigma > 0:
        x_new = gaussian_filter1d(x_new, sigma=sigma, mode='nearest')
        y_new = gaussian_filter1d(y_new, sigma=sigma, mode='nearest')
    return x_new, y_new


def ai_adaptive_optimization(x, y, complexity):
    if len(x) < 2: return x, y
    if complexity < 20:
        num_points, sigma = len(x)*2, 0.5
    elif complexity < 50:
        num_points, sigma = len(x)*2, 0.8
    else:
        num_points, sigma = len(x)*3, 1.0
    return ai_optimized_interpolation(x, y, sigma, num_points)


def dqn_optimized(x, y, delta_alpha, sigma):
    if len(x) < 2: return x, y
    x_rot, y_rot = rotate_coords(x, y, delta_alpha)
    return ai_optimized_interpolation(x_rot, y_rot, sigma)


def hausdorff_fidelity(x1, y1, x2, y2):
    if len(x1) < 2 or len(x2) < 2: return 0.0
    P, Q = np.column_stack([x1, y1]), np.column_stack([x2, y2])
    d = max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])
    span = max(float(np.ptp(x1)), float(np.ptp(y1)), 1.0)
    return max(0.0, 100.0 * (1.0 - d / span))


def curvature_array(x, y):
    if len(x) < 3: return np.array([0.0])
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return np.abs(dx*ddy - dy*ddx) / ((dx**2 + dy**2)**1.5 + 1e-10)


def normalized_smoothness(x, y):
    if len(x) < 3: return 0.0
    curv = curvature_array(x, y)
    length = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))) + 1e-10
    return float(np.mean(curv) * len(x) / length)


# ============================================================================
# Processing
# ============================================================================

def compute_complexity(entry):
    """Compute complexity score from a JSON entry."""
    strokes = entry['strokes']
    n_strokes = len(strokes)
    stroke_types = list(set(s.get('strokeType', '点') for s in strokes))
    return n_strokes * len(stroke_types)


def process_character(entry, agent, methods):
    """Process a single character with specified methods.

    Returns dict with per-method optimized coordinates and metrics.
    """
    char_name = entry['character']
    strokes = entry['strokes']
    n_strokes = len(strokes)
    complexity = compute_complexity(entry)

    if complexity < 20:
        tier = 'Simple'
    elif complexity < 50:
        tier = 'Medium'
    else:
        tier = 'Complex'

    result = {
        'character': char_name,
        'n_strokes': n_strokes,
        'complexity': complexity,
        'tier': tier,
        'methods': {},
    }

    for method in methods:
        method_result = {
            'strokes': [],
            'fidelities': [],
            'smoothness': [],
            'times': [],
        }

        for i, stroke in enumerate(strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2:
                continue

            t0 = time.perf_counter()

            if method == 'baseline':
                x_opt, y_opt = baseline_interpolation(x, y)
            elif method == 'ai_adaptive':
                x_opt, y_opt = ai_adaptive_optimization(x, y, complexity)
            elif method == 'dqn':
                state = agent.build_state(stroke, complexity, n_strokes, i)
                action = agent.select_action(state)
                da, sig = agent.decode_action(action)
                x_opt, y_opt = dqn_optimized(x, y, da, sig)
                # Store DQN decision info
                method_result.setdefault('dqn_params', []).append({
                    'stroke_idx': i,
                    'action': action,
                    'delta_alpha': da,
                    'sigma': sig,
                })

            elapsed_ms = (time.perf_counter() - t0) * 1000

            fid = hausdorff_fidelity(x, y, x_opt, y_opt)
            sm = normalized_smoothness(x_opt, y_opt)

            method_result['strokes'].append({
                'stroke_idx': i,
                'stroke_type': stroke.get('strokeType', '点'),
                'original_points': len(x),
                'optimized_points': len(x_opt),
                'coordinates': list(zip(x_opt.tolist(), y_opt.tolist())),
                'fidelity': round(fid, 2),
                'smoothness': round(sm, 4),
                'time_ms': round(elapsed_ms, 3),
            })
            method_result['fidelities'].append(fid)
            method_result['smoothness'].append(sm)
            method_result['times'].append(elapsed_ms)

        # Aggregate
        if method_result['fidelities']:
            method_result['avg_fidelity'] = round(
                float(np.mean(method_result['fidelities'])), 2)
            method_result['avg_smoothness'] = round(
                float(np.mean(method_result['smoothness'])), 4)
            method_result['total_time_ms'] = round(
                float(np.sum(method_result['times'])), 3)

        # Remove raw lists from output (keep aggregated)
        del method_result['fidelities']
        del method_result['smoothness']
        del method_result['times']

        result['methods'][method] = method_result

    return result


def visualize_results(results, output_dir, max_chars=20):
    """Generate visualization of processed characters."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    cjk_candidates = [
        'SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
        'Source Han Sans SC', 'Droid Sans Fallback']
    available = {f.name for f in fm.fontManager.ttflist}
    chosen_fonts = [f for f in cjk_candidates if f in available]
    plt.rcParams['font.sans-serif'] = chosen_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    import warnings
    warnings.filterwarnings('ignore', message='Glyph .* missing from')

    TIER_COLORS = {'Simple': '#27AE60', 'Medium': '#E67E22', 'Complex': '#C0392B'}

    # Determine which methods are present
    sample = results[0]
    methods = list(sample['methods'].keys())
    n_methods = len(methods)
    method_labels = {
        'baseline': 'Baseline', 'ai_adaptive': 'AI Adaptive', 'dqn': 'DQN'}

    chars = results[:max_chars]
    n_chars = len(chars)
    ncols = min(5, n_chars)
    nrows = math.ceil(n_chars / ncols)

    for method in methods:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.2, nrows * 3.2))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx, char_result in enumerate(chars):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            char_name = char_result['character']
            tier = char_result['tier']
            n_strokes = char_result['n_strokes']
            complexity = char_result['complexity']
            m_data = char_result['methods'].get(method, {})

            for si, s in enumerate(m_data.get('strokes', [])):
                coords = s['coordinates']
                xp = [c[0] for c in coords]
                yp = [-c[1] for c in coords]
                color = plt.cm.Dark2(si / max(1, n_strokes - 1))
                ax.plot(xp, yp, color=color, lw=1.8, alpha=0.85)
                ax.plot(xp[0], yp[0], 'o', color=color, ms=3, alpha=0.6)

            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(TIER_COLORS.get(tier, '#888'))
                spine.set_linewidth(2.5)

            fid = m_data.get('avg_fidelity', 0)
            ax.set_title(
                f'{char_name} [{tier}]\n'
                f'C={complexity}, fid={fid:.1f}%',
                fontsize=8, fontweight='bold', pad=4)

        # Hide unused subplots
        for idx in range(n_chars, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        label = method_labels.get(method, method)
        fig.suptitle(f'Optimized Results — {label} (n={n_chars})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        path = os.path.join(output_dir, f'visualize_{method}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stroke2Font DQN Inference — process characters '
                    'with trained model')
    parser.add_argument('--input', '-i', type=str,
                        default='data/paper-1-180xk.json',
                        help='Input JSON file (default: paper-1-180xk.json)')
    parser.add_argument('--model', '-m', type=str,
                        default='results/dqn_model.pt',
                        help='Trained DQN model path')
    parser.add_argument('--output', '-o', type=str,
                        default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['baseline', 'ai_adaptive', 'dqn', 'all'],
                        help='Optimization method (default: all)')
    parser.add_argument('--limit', '-n', type=int, default=0,
                        help='Only process first N characters (0=all)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization PNGs')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for inference')
    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    os.chdir(project_dir)

    # Setup methods
    if args.method == 'all':
        methods = ['baseline', 'ai_adaptive', 'dqn']
    else:
        methods = [args.method]

    # Load model
    agent = DQNInference(args.model, args.device)

    # Load data
    print(f"\nLoading: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Filter bad entries
    entries = []
    for entry in raw_data:
        strokes = entry.get('strokes', [])
        n_strokes = len(strokes)
        n_points = sum(
            s.get('pointCount', len(s.get('coordinates', [])))
            for s in strokes)
        if n_strokes <= 1 and n_points <= 1:
            continue
        entries.append(entry)

    if args.limit > 0:
        entries = entries[:args.limit]
    print(f"Characters to process: {len(entries)}")

    # Process
    os.makedirs(args.output, exist_ok=True)
    print(f"\nProcessing with methods: {methods}")
    t_start = time.time()

    results = []
    for i, entry in enumerate(entries):
        result = process_character(entry, agent, methods)
        results.append(result)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i+1}/{len(entries)} — {entry['character']}")

    elapsed = time.time() - t_start
    print(f"\nDone: {len(results)} characters in {elapsed:.1f}s "
          f"({elapsed/len(results)*1000:.1f}ms/char)")

    # Save results JSON
    json_path = os.path.join(args.output, 'optimized_results.json')

    def to_serializable(obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=to_serializable)
    print(f"\n[OK] Results: {json_path}")

    # Summary
    summary_lines = []
    summary_lines.append(f"Stroke2Font DQN Inference Summary")
    summary_lines.append(f"=" * 50)
    summary_lines.append(f"Input:  {args.input} ({len(entries)} chars)")
    summary_lines.append(f"Model:  {args.model}")
    summary_lines.append(f"Device: {args.device}")
    summary_lines.append(f"Time:   {elapsed:.1f}s")
    summary_lines.append(f"")

    for method in methods:
        fids = [r['methods'][method]['avg_fidelity']
                for r in results if method in r['methods']
                and 'avg_fidelity' in r['methods'][method]]
        sms = [r['methods'][method]['avg_smoothness']
               for r in results if method in r['methods']
               and 'avg_smoothness' in r['methods'][method]]
        tms = [r['methods'][method]['total_time_ms']
               for r in results if method in r['methods']
               and 'total_time_ms' in r['methods'][method]]

        label = {'baseline': 'Baseline', 'ai_adaptive': 'AI Adaptive',
                 'dqn': 'DQN (Ours)'}.get(method, method)
        summary_lines.append(f"  {label}:")
        summary_lines.append(
            f"    Fidelity:   {np.mean(fids):.1f}% ± {np.std(fids):.1f}%")
        summary_lines.append(
            f"    Smoothness: {np.mean(sms):.4f}")
        summary_lines.append(
            f"    Avg time:   {np.mean(tms):.2f}ms")
        summary_lines.append(f"")

    summary_text = '\n'.join(summary_lines)
    print(f"\n{summary_text}")

    summary_path = os.path.join(args.output, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"[OK] Summary: {summary_path}")

    # Visualization
    if args.visualize:
        print(f"\nGenerating visualizations...")
        visualize_results(results, args.output)

    print(f"\n{'='*50}")
    print(f"All output saved to: {os.path.abspath(args.output)}/")


if __name__ == '__main__':
    main()
