#!/usr/bin/env python3
"""Generate dataset overview: first 20 characters with Dark2 colormap."""

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
    'Simple':  '#27AE60',   # green
    'Medium':  '#E67E22',   # orange
    'Complex': '#C0392B',   # red
}


def extract_coords(stroke):
    coords = stroke.get('coordinates', [])
    x = np.array([float(p[0]) for p in coords if len(p) >= 2])
    y = np.array([float(p[1]) for p in coords if len(p) >= 2])
    return x, y


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

        # Draw each stroke
        for i, stroke in enumerate(strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2:
                continue
            color = plt.cm.Dark2(i / max(1, n_strokes - 1))
            ax.plot(x, -np.array(y), color=color, lw=1.8, alpha=0.85)
            # Mark start point
            ax.plot(x[0], -y[0], 'o', color=color, ms=3, alpha=0.6)

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Colored border by complexity tier
        border_color = TIER_COLORS[tier]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        # Title: character name + info
        ax.set_title(
            f'#{idx+1} {char_name}  [{tier}]\n'
            f'{n_strokes} strokes, C={complexity}',
            fontsize=9, fontweight='bold', pad=4)

    # Legend for tier colors
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
        'Dataset Overview — Test Set Characters (Stroke Element Trajectories)',
        fontsize=14, fontweight='bold', y=0.98)
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=10,
               frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = 'sample_20_characters.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
