#!/usr/bin/env python3
"""Generate dataset overview: first 20 characters — AI Adaptive optimized."""

import json
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── CJK font ────────────────────────────────────────────────────────────
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

TIER_COLORS = {
    'Simple':  '#27AE60',
    'Medium':  '#E67E22',
    'Complex': '#C0392B',
}


def extract_coords(stroke):
    coords = stroke.get('coordinates', [])
    x = np.array([float(p[0]) for p in coords if len(p) >= 2])
    y = np.array([float(p[1]) for p in coords if len(p) >= 2])
    return x, y


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


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    with open('data/paper-1-180xk.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Test set: entries 150-169 (first 20 of 30 holdout chars)
    entries = raw_data[150:170]

    fig, axes = plt.subplots(4, 5, figsize=(16, 13))

    for idx, entry in enumerate(entries):
        row, col = divmod(idx, 5)
        ax = axes[row, col]

        char_name = entry['character']
        strokes = entry['strokes']
        n_strokes = len(strokes)
        stroke_types = list(set(s.get('strokeType', '点') for s in strokes))
        complexity = n_strokes * len(stroke_types)

        if complexity < 20:
            tier = 'Simple'
        elif complexity < 50:
            tier = 'Medium'
        else:
            tier = 'Complex'

        # Draw each stroke (AI Adaptive optimized)
        for i, stroke in enumerate(strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2:
                continue
            x_opt, y_opt = ai_adaptive_optimization(x, y, complexity)
            color = plt.cm.Dark2(i / max(1, n_strokes - 1))
            ax.plot(x_opt, -np.array(y_opt), color=color, lw=1.8, alpha=0.85)
            ax.plot(x_opt[0], -y_opt[0], 'o', color=color, ms=3, alpha=0.6)

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        border_color = TIER_COLORS[tier]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        ax.set_title(
            f'#{idx+1} {char_name}  [{tier}]\n'
            f'{n_strokes} strokes, C={complexity}',
            fontsize=9, fontweight='bold', pad=4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor=TIER_COLORS['Simple'],
              linewidth=2.5, label='Simple (C<20)'),
        Patch(facecolor='white', edgecolor=TIER_COLORS['Medium'],
              linewidth=2.5, label='Medium (20≤C<50)'),
        Patch(facecolor='white', edgecolor=TIER_COLORS['Complex'],
              linewidth=2.5, label='Complex (C≥50)'),
    ]

    fig.suptitle(
        'Dataset Overview — Test Set Characters (AI Adaptive Optimized)',
        fontsize=14, fontweight='bold', y=0.98)
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=10,
               frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = 'sample_20_characters_optimized.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
