#!/usr/bin/env python3
"""Regenerate stroke_comparison.png with a specific character ('啊')."""

import json
import os
import sys
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CJK font setup ──────────────────────────────────────────────────────
cjk_candidates = [
    'SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei',
    'Source Han Sans SC', 'Droid Sans Fallback']
available = {f.name for f in fm.fontManager.ttflist}
chosen = [f for f in cjk_candidates if f in available]
plt.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', message='Glyph .* missing from')

# ── Constants ────────────────────────────────────────────────────────────
STROKE_TYPES = ['横向笔画', '纵向笔画', '撇捺笔画', '折转笔画', '复合笔画', '点']
TARGET_CHAR = '啊'

# ── Pipeline functions (from main script) ────────────────────────────────

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

def curvature_variance(x, y):
    return float(np.var(curvature_array(x, y)))

# ── QNetwork ─────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim=10, action_dim=20, hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # Load data
    with open('data/paper-1-180xk.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Find target character
    char_entry = None
    for entry in raw_data:
        if entry['character'] == TARGET_CHAR:
            char_entry = entry
            break
    if char_entry is None:
        print(f"Character '{TARGET_CHAR}' not found!")
        return

    strokes = char_entry['strokes']
    n_strokes = len(strokes)
    stroke_type_set = list(set(s.get('strokeType', '点') for s in strokes))
    complexity = n_strokes * len(stroke_type_set)
    print(f"Character: {TARGET_CHAR}, strokes={n_strokes}, complexity={complexity}")

    # Load DQN model
    device = torch.device('cpu')
    checkpoint = torch.load('results/dqn_model.pt', map_location=device,
                            weights_only=True)
    q_cfg = checkpoint['config']
    q_net = QNetwork(q_cfg['state_dim'], q_cfg['action_dim'],
                     q_cfg['hidden1'], q_cfg['hidden2'])
    q_net.load_state_dict(checkpoint['q_net_state_dict'])
    q_net.eval()

    # Action lookup
    delta_alpha_values = [-10, -5, 0, 5, 10]
    sigma_values = [0.5, 0.8, 1.0, 1.5]
    actions_lut = [(da, s) for da in delta_alpha_values for s in sigma_values]

    def build_state(stroke, stroke_idx):
        stype = stroke.get('strokeType', '点')
        tidx = STROKE_TYPES.index(stype) if stype in STROKE_TYPES else 5
        oh = [0.0]*6; oh[tidx] = 1.0
        c_norm = min(complexity / 100.0, 1.0)
        coords = stroke.get('coordinates', [])
        n_pts = len(coords)
        if n_pts >= 2:
            import math
            length = sum(math.sqrt((coords[i][0]-coords[i-1][0])**2 +
                                   (coords[i][1]-coords[i-1][1])**2)
                         for i in range(1, n_pts))
            density = min(n_pts / (length + 1e-10), 1.0)
            length_norm = min(length / 100.0, 1.0)
        else:
            density, length_norm = 0.0, 0.0
        budget = 1.0 - stroke_idx / max(n_strokes, 1)
        return np.array(oh + [c_norm, density, length_norm, budget], dtype=np.float32)

    def dqn_select(stroke, stroke_idx):
        state = build_state(stroke, stroke_idx)
        with torch.no_grad():
            q = q_net(torch.FloatTensor(state).unsqueeze(0))
            action = int(q.argmax(dim=1).item())
        return actions_lut[action]

    # ── Generate figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    method_specs = [
        ('Original', None),
        ('Baseline', 'baseline'),
        ('AI Adaptive', 'ai_adaptive'),
        ('DQN', 'dqn'),
    ]

    # Top row: full character
    for col, (name, method) in enumerate(method_specs):
        ax = axes[0, col]
        for i, stroke in enumerate(strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2: continue
            if method is None:
                xp, yp = x, y
            elif method == 'baseline':
                xp, yp = baseline_interpolation(x, y)
            elif method == 'ai_adaptive':
                xp, yp = ai_adaptive_optimization(x, y, complexity)
            elif method == 'dqn':
                da, sig = dqn_select(stroke, i)
                xp, yp = dqn_optimized(x, y, da, sig)
            else:
                xp, yp = x, y
            color = plt.cm.Dark2(i / max(1, n_strokes - 1))
            ax.plot(xp, -np.array(yp), color=color, lw=2, alpha=0.8)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n{TARGET_CHAR}', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom row: single stroke detail
    stroke_idx = min(2, len(strokes) - 1)
    stroke = strokes[stroke_idx]
    x_orig, y_orig = extract_coords(stroke)

    if len(x_orig) >= 2:
        x_base, y_base = baseline_interpolation(x_orig, y_orig)
        x_adap, y_adap = ai_adaptive_optimization(x_orig, y_orig, complexity)
        da, sig = dqn_select(stroke, stroke_idx)
        x_dqn, y_dqn = dqn_optimized(x_orig, y_orig, da, sig)

        versions = [
            ('Original', x_orig, y_orig),
            ('Baseline', x_base, y_base),
            ('AI Adaptive', x_adap, y_adap),
            ('DQN', x_dqn, y_dqn),
        ]
        detail_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
        for col, (name, xp, yp) in enumerate(versions):
            ax = axes[1, col]
            ax.plot(xp, -np.array(yp), '-', color=detail_colors[col], lw=2)
            if col == 0:
                ax.plot(xp, -np.array(yp), 'o',
                        color=detail_colors[col], ms=3, alpha=0.5)
            cv = curvature_variance(xp, yp)
            fid = hausdorff_fidelity(x_orig, y_orig, xp, yp) if col > 0 else 100.0
            ax.set_title(f'{name}\nκ_var={cv:.4f}, fid={fid:.1f}%', fontsize=9)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)

    plt.suptitle(
        f'Stroke Comparison — "{TARGET_CHAR}" '
        f'({n_strokes} strokes, C={complexity:.0f})',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = 'stroke_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.abspath(out_path)}")

    # Also save to results/
    out_path2 = 'results/fig_stroke_comparison.png'
    import shutil
    shutil.copy2(out_path, out_path2)
    print(f"Saved: {os.path.abspath(out_path2)}")


if __name__ == '__main__':
    main()
