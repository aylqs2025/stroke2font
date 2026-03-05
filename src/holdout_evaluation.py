"""
Holdout Evaluation: 30 Unseen Chinese Characters (Real Heiti Stroke Data)
Stroke2Font — Generalization Validation

Data source: data/char30ht.txt  (characters 151-180, same format as ch150)

Pipeline:
  1. Parse char30ht.txt  →  build dataset JSON
  2. Run three-method evaluation (Baseline / AI Optimized / AI Adaptive)
  3. Print paper-ready tables
  4. Save holdout_evaluation.png (300 DPI)
"""

import json
import math
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive — saves PNG without display window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR  = r"d:\APP\python\mypaper-program\algorithms-1\data"
OUT_DIR   = r"d:\APP\python\mypaper-program\algorithms-1"
CHAR30_HT   = f"{DATA_DIR}/char30ht.txt"      # Heiti  (comma-sep)
CHAR30_XK   = f"{DATA_DIR}/char30xk.txt"      # Xingkai (tab-sep)
CHAR30_FILE = CHAR30_XK                        # ← active file
FONT_LABEL  = "Xingkai"                        # label for output
SOURCE_TAG  = "char30xk_real_xingkai"
OUT_JSON    = f"{DATA_DIR}/hanzi_dataset_holdout_30_xk.json"
OUT_PNG     = f"{OUT_DIR}/holdout_evaluation_xk.png"

# Known results from the primary 150-character experiment
PRIMARY = {
    'baseline':     {'avg_fidelity': 60.9, 'std_fidelity': 8.2,
                     'avg_smoothness': 0.174, 'std_smoothness': 0.0, 'avg_time': 9.38},
    'ai_optimized': {'avg_fidelity': 60.8, 'std_fidelity': 8.1,
                     'avg_smoothness': 0.339, 'std_smoothness': 0.0, 'avg_time': 12.14},
    'ai_adaptive':  {'avg_fidelity': 65.2, 'std_fidelity': 7.5,
                     'avg_smoothness': 9.59,  'std_smoothness': 0.0, 'avg_time': 12.18},
}

# ============================================================================
# 1.  PARSE char30ht.txt
# ============================================================================

def parse_raw_stroke_data(raw: str) -> List[List[List[int]]]:
    """
    Parse ch150-format raw string into list-of-strokes (absolute coords).
    Delimiter:  -64,0  = new stroke;   -64,-64 = end of character.
    """
    parts = raw.strip().split(',')
    values = []
    for p in parts:
        p = p.strip()
        if p in ('', 'NaN'):
            continue
        try:
            values.append(int(p))
        except ValueError:
            try:
                values.append(float(p))
            except ValueError:
                pass

    if len(values) < 2:
        return []

    strokes, current = [], []
    i = 1                           # skip leading total-indicator
    while i < len(values):
        v = values[i]
        if v == -64 and i + 1 < len(values):
            nxt = values[i + 1]
            if nxt == 0:            # stroke delimiter
                if current:
                    strokes.append(current)
                    current = []
                i += 2
                continue
            if nxt == -64:          # end marker
                if current:
                    strokes.append(current)
                break
        if i + 1 < len(values) and v != -64:
            x, y = values[i], values[i + 1]
            if x != -64:
                current.append([x, y])
            i += 2
        else:
            i += 1

    return strokes


def classify_stroke_type(coords: List[List[int]]) -> str:
    if len(coords) < 2:
        return "点"
    start, end = coords[0], coords[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    dir_changes = sum(
        1 for i in range(1, len(coords) - 1)
        if (coords[i][0]-coords[i-1][0])*(coords[i+1][0]-coords[i][0]) < 0
        or (coords[i][1]-coords[i-1][1])*(coords[i+1][1]-coords[i][1]) < 0
    )
    if dir_changes >= 2:
        return "复合笔画"
    ang = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0
    if   -30  <= ang <=  30:  return "横向笔画"
    elif  60  <= ang <= 120:  return "纵向笔画"
    elif -120 <= ang <= -60:  return "纵向笔画"
    elif  30  < ang <   60:   return "撇捺笔画"
    elif -60  < ang <  -30:   return "撇捺笔画"
    elif dir_changes >= 1:    return "折转笔画"
    else:                     return "横向笔画"


def to_absolute(rel_strokes: List[List[List[int]]]) -> List[List[List[int]]]:
    """Convert relative-offset strokes to absolute coordinates."""
    abs_strokes = []
    cx, cy = 0, 0
    for stroke in rel_strokes:
        abs_s = []
        for dx, dy in stroke:
            cx += dx
            cy += dy
            abs_s.append([cx, cy])
        abs_strokes.append(abs_s)
    return abs_strokes


def _build_entries(char: str, rel_strokes: List, raw_str: str, source: str) -> Dict:
    """Shared helper: build dataset entry from parsed relative strokes.

    NOTE: coordinates are stored as RAW RELATIVE OFFSETS (not cumulative absolute)
    to match the format of hanzi_dataset_150_chars.json generated by generate_dataset.py.
    performance_evaluation.py reads and interpolates these offsets directly.
    """
    strokes, total_pts = [], 0
    for idx, coords in enumerate(rel_strokes):  # keep relative offsets
        if not coords:
            continue
        stype = classify_stroke_type(coords)
        xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
        bbox = {"minX": min(xs), "minY": min(ys),
                "maxX": max(xs), "maxY": max(ys),
                "width": max(xs)-min(xs), "height": max(ys)-min(ys)}
        length = sum(
            math.sqrt((coords[i][0]-coords[i-1][0])**2 +
                      (coords[i][1]-coords[i-1][1])**2)
            for i in range(1, len(coords))
        )
        strokes.append({
            "strokeIndex": idx + 1,
            "strokeType":  stype,
            "pointCount":  len(coords),
            "coordinates": coords,
            "boundingBox": bbox,
            "length":      round(length, 2)
        })
        total_pts += len(coords)
    if not strokes:
        return None
    return {
        "character": char,
        "metadata": {"totalStrokes": len(strokes), "totalPoints": total_pts,
                     "source": source, "version": "1.0",
                     "format": "hanzi-coordinate-model"},
        "rawData": raw_str,
        "strokes": strokes
    }


def load_char30ht(path: str) -> List[Dict]:
    """Parse char30ht.txt (comma-sep values, 3-col TSV with quotes)."""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            char    = parts[1].strip()
            raw_str = parts[2].strip().strip('"')

            rel_strokes = parse_raw_stroke_data(raw_str)
            if not rel_strokes:
                print(f"  [WARN] No strokes for '{char}', skipping.")
                continue

            entry = _build_entries(char, rel_strokes, raw_str, "char30ht_real_heiti")
            if entry:
                dataset.append(entry)
    return dataset


def load_char30xk(path: str) -> List[Dict]:
    """Parse char30xk.txt (tab-sep values: char TAB total TAB v1 TAB v2 ...)."""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            char = parts[0].strip()
            if not char:
                continue
            # values start from col 1 (total_indicator then coords)
            raw_str = ','.join(p for p in parts[1:] if p.strip() not in ('', 'NaN'))
            rel_strokes = parse_raw_stroke_data(raw_str)
            if not rel_strokes:
                print(f"  [WARN] No strokes for '{char}', skipping.")
                continue
            entry = _build_entries(char, rel_strokes, raw_str, "char30xk_real_xingkai")
            if entry:
                dataset.append(entry)
    return dataset




# ============================================================================
# 2.  EVALUATION PIPELINE  (same logic as performance_evaluation.py)
# ============================================================================

def extract_xy(stroke: Dict) -> Tuple[np.ndarray, np.ndarray]:
    coords = stroke['coordinates']
    x = np.array([float(p[0]) for p in coords])
    y = np.array([float(p[1]) for p in coords])
    return x, y


def hausdorff_similarity(x1, y1, x2, y2) -> float:
    if len(x1) < 2 or len(x2) < 2:
        return 0.0
    P = np.column_stack([x1, y1])
    Q = np.column_stack([x2, y2])
    d = max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])
    span = max(float(np.ptp(x1)), float(np.ptp(y1)), 1.0)
    return max(0.0, 100.0 * (1.0 - d / span))


def smoothness(x, y) -> float:
    if len(x) < 3:
        return 0.0
    dx, dy   = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curv = np.abs(dx*ddy - dy*ddx) / ((dx**2 + dy**2)**1.5 + 1e-10)
    length = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))) + 1e-10
    return float(np.mean(curv) * len(x) / length)


def baseline_interp(x, y):
    if len(x) < 2: return x, y
    t  = np.linspace(0, 1, len(x))
    t2 = np.linspace(0, 1, max(len(x), min(50, len(x)*3)))
    return interp1d(t, x)(t2), interp1d(t, y)(t2)


def ai_optimized_interp(x, y, sigma=1.0):
    if len(x) < 2: return x, y
    t    = np.linspace(0, 1, len(x))
    t2   = np.linspace(0, 1, max(len(x), min(50, len(x)*3)))
    kind = 'cubic' if len(x) >= 4 else 'linear'
    xn   = interp1d(t, x, kind=kind)(t2)
    yn   = interp1d(t, y, kind=kind)(t2)
    if sigma > 0:
        xn = gaussian_filter1d(xn, sigma=sigma, mode='nearest')
        yn = gaussian_filter1d(yn, sigma=sigma, mode='nearest')
    return xn, yn


def ai_adaptive_interp(x, y, complexity: float):
    if len(x) < 2: return x, y
    sigma = 0.5 if complexity < 20 else (0.8 if complexity < 50 else 1.0)
    return ai_optimized_interp(x, y, sigma)


def evaluate_char(entry: Dict, method: str) -> Dict:
    strokes    = entry['strokes']
    stypes     = set(s['strokeType'] for s in strokes)
    complexity = len(strokes) * len(stypes)

    sims, smths, times = [], [], []
    for stroke in strokes:
        x, y = extract_xy(stroke)
        if len(x) < 2:
            continue
        t0 = time.perf_counter()
        if method == 'baseline':
            xo, yo = baseline_interp(x, y)
        elif method == 'ai_optimized':
            xo, yo = ai_optimized_interp(x, y, sigma=1.0)
        else:
            xo, yo = ai_adaptive_interp(x, y, complexity)
        times.append((time.perf_counter() - t0) * 1000)
        sims.append(hausdorff_similarity(x, y, xo, yo))
        smths.append(smoothness(xo, yo))

    return {
        'character':       entry['character'],
        'stroke_count':    len(strokes),
        'complexity_score': complexity,
        'stroke_types':    list(stypes),
        'avg_similarity':  float(np.mean(sims))  if sims  else 0.0,
        'avg_smoothness':  float(np.mean(smths)) if smths else 0.0,
        'generation_time': float(np.sum(times)),
    }


def run_evaluation(dataset: List[Dict]) -> Dict:
    methods = ['baseline', 'ai_optimized', 'ai_adaptive']
    results = {m: [] for m in methods}

    print("\n" + "="*70)
    print(f"HOLDOUT EVALUATION  —  30 UNSEEN CHARACTERS  [{FONT_LABEL}]")
    print("="*70)

    for method in methods:
        print(f"\n  [{method.upper()}]")
        for entry in dataset:
            r = evaluate_char(entry, method)
            results[method].append(r)
            print(f"    {r['character']}  strokes={r['stroke_count']:2d}  "
                  f"C={r['complexity_score']:3.0f}  "
                  f"fidelity={r['avg_similarity']:5.1f}%  "
                  f"time={r['generation_time']:.3f}ms")
    return results


def summarize(results: Dict) -> Dict:
    out = {}
    for method, rlist in results.items():
        sims  = [r['avg_similarity']  for r in rlist]
        smths = [r['avg_smoothness']  for r in rlist]
        times = [r['generation_time'] for r in rlist]
        out[method] = {
            'avg_fidelity':   float(np.mean(sims)),
            'std_fidelity':   float(np.std(sims)),
            'avg_smoothness': float(np.mean(smths)),
            'std_smoothness': float(np.std(smths)),
            'avg_time':       float(np.mean(times)),
            'std_time':       float(np.std(times)),
        }
    return out


# ============================================================================
# 3.  COMPLEXITY-STRATIFIED ANALYSIS
# ============================================================================

def stratified(results: Dict) -> Dict:
    tiers = {
        'Simple (C<20)':    (0,  20),
        'Medium (20≤C<50)': (20, 50),
        'Complex (C≥50)':   (50, 9999),
    }
    out = {t: {'n': 0, 'base': [], 'adapt': []} for t in tiers}
    for i, entry in enumerate(results['baseline']):
        c = entry['complexity_score']
        for t, (lo, hi) in tiers.items():
            if lo <= c < hi:
                out[t]['base'].append(entry['avg_similarity'])
                out[t]['adapt'].append(results['ai_adaptive'][i]['avg_similarity'])
                out[t]['n'] += 1
                break
    return {
        t: {
            'n':           d['n'],
            'base_mean':   float(np.mean(d['base']))  if d['base']  else 0.0,
            'adapt_mean':  float(np.mean(d['adapt'])) if d['adapt'] else 0.0,
            'improvement': float(np.mean(d['adapt']) - np.mean(d['base']))
                           if d['base'] else 0.0
        }
        for t, d in out.items()
    }


# ============================================================================
# 4.  PAPER TABLES
# ============================================================================

def print_tables(summary: Dict, strat: Dict, results: Dict):
    W = 72

    # -- Table A: overall --
    print("\n" + "="*W)
    print("TABLE A  Overall Performance — Holdout Set (n=30)")
    print("="*W)
    print(f"  {'Metric':<28} {'Baseline':>12} {'AI Optimized':>14} {'AI Adaptive':>13}")
    print("-"*W)
    for label, key, fmt, suf in [
        ("Trajectory Fidelity (%)",  'avg_fidelity',   '.1f', '%'),
        ("Avg Curvature (smoothness)",'avg_smoothness', '.4f', '' ),
        ("Generation Time (ms)",     'avg_time',       '.2f', '' ),
    ]:
        vals = [summary[m][key]                         for m in ['baseline','ai_optimized','ai_adaptive']]
        stds = [summary[m][key.replace('avg_','std_')]  for m in ['baseline','ai_optimized','ai_adaptive']]
        cells = [f"{v:{fmt}}{suf}±{s:{fmt}}" for v, s in zip(vals, stds)]
        print(f"  {label:<28} {cells[0]:>12} {cells[1]:>14} {cells[2]:>13}")
    bsl = summary['baseline']['avg_fidelity']
    opt = summary['ai_optimized']['avg_fidelity']
    ada = summary['ai_adaptive']['avg_fidelity']
    print(f"  {'Fidelity improvement':<28} {'—':>12} {opt-bsl:>+13.1f}% {ada-bsl:>+12.1f}%")

    # -- Table B: complexity tiers --
    print("\n" + "="*W)
    print("TABLE B  Complexity-Stratified Results — Holdout")
    print("="*W)
    print(f"  {'Tier':<24} {'n':>4} {'Baseline':>10} {'AI Adaptive':>12} {'Δ':>10}")
    print("-"*W)
    for t, d in strat.items():
        print(f"  {t:<24} {d['n']:>4} {d['base_mean']:>9.1f}% "
              f"{d['adapt_mean']:>11.1f}% {d['improvement']:>+9.1f}%")

    # -- Table C: generalization gap --
    pb = PRIMARY['baseline']['avg_fidelity'];   ps = PRIMARY['baseline']['std_fidelity']
    pa = PRIMARY['ai_adaptive']['avg_fidelity']; qs = PRIMARY['ai_adaptive']['std_fidelity']
    hb = summary['baseline']['avg_fidelity'];   hs = summary['baseline']['std_fidelity']
    ha = summary['ai_adaptive']['avg_fidelity']; ht = summary['ai_adaptive']['std_fidelity']
    print("\n" + "="*W)
    print("TABLE C  Generalization Comparison")
    print("="*W)
    print(f"  {'Split':<24} {'n':>5} {'Baseline':>14} {'AI Adaptive':>14} {'Δ':>8}")
    print("-"*W)
    print(f"  {'Primary (opt set)':<24} {'150':>5}  {pb:.1f}%±{ps:.1f}%   {pa:.1f}%±{qs:.1f}%  {pa-pb:>+6.1f}%")
    print(f"  {'Holdout (unseen)':<24} {'30':>5}  {hb:.1f}%±{hs:.1f}%   {ha:.1f}%±{ht:.1f}%  {ha-hb:>+6.1f}%")
    print(f"\n  Generalization gap (primary - holdout):  "
          f"Baseline {abs(pb-hb):.1f}%  |  AI Adaptive {abs(pa-ha):.1f}%")
    print()


# ============================================================================
# 5.  VISUALIZATION
# ============================================================================

def plot_results(results: Dict, summary: Dict, strat: Dict):
    methods = ['baseline', 'ai_optimized', 'ai_adaptive']
    labels  = ['Baseline\n(Linear)', 'AI Optimized\n(Cubic+Smooth)', 'AI Adaptive\n(Ours)']
    colors  = ['#E74C3C', '#3498DB', '#2ECC71']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ── (A) Fidelity bars ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    means = [summary[m]['avg_fidelity'] for m in methods]
    stds  = [summary[m]['std_fidelity']  for m in methods]
    bars  = ax.bar(range(3), means, yerr=stds, color=colors, capsize=5,
                   alpha=0.85, edgecolor='black')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Trajectory Fidelity (%)', fontsize=9)
    ax.set_title('(A) Fidelity — Holdout (n=30)', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # improvement annotation
    imp = means[2] - means[0]
    ax.annotate(f'+{imp:.1f}%\nimprovement',
                xy=(2, means[2]/2), fontsize=8, ha='center',
                color='darkgreen', fontweight='bold')

    # ── (B) Generalization comparison ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    groups = ['Baseline', 'AI Adaptive']
    p_vals = [PRIMARY['baseline']['avg_fidelity'],   PRIMARY['ai_adaptive']['avg_fidelity']]
    p_stds = [PRIMARY['baseline']['std_fidelity'],   PRIMARY['ai_adaptive']['std_fidelity']]
    h_vals = [summary['baseline']['avg_fidelity'],   summary['ai_adaptive']['avg_fidelity']]
    h_stds = [summary['baseline']['std_fidelity'],   summary['ai_adaptive']['std_fidelity']]
    x = np.arange(2); w = 0.35
    b1 = ax.bar(x-w/2, p_vals, w, yerr=p_stds, label='Primary (n=150)',
                color=['#E74C3C','#2ECC71'], alpha=0.70, capsize=4, edgecolor='black')
    b2 = ax.bar(x+w/2, h_vals, w, yerr=h_stds, label='Holdout (n=30)',
                color=['#C0392B','#27AE60'], alpha=0.90, capsize=4, edgecolor='black', hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel('Trajectory Fidelity (%)', fontsize=9)
    ax.set_title('(B) Generalization: Primary vs Holdout', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 85); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    for xp, yp in zip(np.concatenate([x-w/2, x+w/2]),
                       np.concatenate([p_vals, h_vals])):
        ax.text(xp, yp+0.4, f'{yp:.1f}%', ha='center', va='bottom', fontsize=8)

    # ── (C) Generation time ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    t_means = [summary[m]['avg_time'] for m in methods]
    t_stds  = [summary[m]['std_time']  for m in methods]
    bars3 = ax.bar(range(3), t_means, yerr=t_stds, color=colors, capsize=5,
                   alpha=0.85, edgecolor='black')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Generation Time (ms)', fontsize=9)
    ax.set_title('(C) Computational Efficiency', fontsize=10, fontweight='bold')
    ax.axhline(15, color='red', linestyle='--', lw=1.2, label='15 ms target')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars3, t_means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    # ── (D) Complexity-stratified bars ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    tkeys  = list(strat.keys())
    ns     = [strat[t]['n']          for t in tkeys]
    b_vals = [strat[t]['base_mean']  for t in tkeys]
    a_vals = [strat[t]['adapt_mean'] for t in tkeys]
    imps   = [strat[t]['improvement']for t in tkeys]
    x = np.arange(len(tkeys)); w = 0.32
    ax.bar(x-w/2, b_vals, w, color='#E74C3C', alpha=0.8, label='Baseline', edgecolor='black')
    b2 = ax.bar(x+w/2, a_vals, w, color='#2ECC71', alpha=0.8, label='AI Adaptive', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(n={ns[i]})' for i, t in enumerate(tkeys)], fontsize=9)
    ax.set_ylabel('Trajectory Fidelity (%)', fontsize=9)
    ax.set_title('(D) Fidelity by Complexity Tier', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar, imp in zip(b2, imps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'+{imp:.1f}%', ha='center', va='bottom', fontsize=9,
                color='darkgreen', fontweight='bold')

    # ── (E) Complexity vs Fidelity scatter ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for method, color, marker, lab in [
        ('baseline',    '#E74C3C', 'o', 'Baseline'),
        ('ai_adaptive', '#2ECC71', '^', 'AI Adaptive'),
    ]:
        cx = [r['complexity_score'] for r in results[method]]
        fy = [r['avg_similarity']   for r in results[method]]
        ax.scatter(cx, fy, c=color, alpha=0.75, s=65,
                   edgecolors='black', lw=0.5, marker=marker, label=lab)
    ax.set_xlabel('Complexity Score (strokes × types)', fontsize=9)
    ax.set_ylabel('Trajectory Fidelity (%)', fontsize=9)
    ax.set_title('(E) Complexity vs Fidelity', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── (F) Per-character improvement bar chart ─────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    chars  = [r['character']      for r in results['baseline']]
    base_f = [r['avg_similarity'] for r in results['baseline']]
    ada_f  = [r['avg_similarity'] for r in results['ai_adaptive']]
    delta  = [a - b for a, b in zip(ada_f, base_f)]
    mean_d = float(np.mean(delta))

    bar_colors = ['#27AE60' if v >= 0 else '#C0392B' for v in delta]
    x_pos = np.arange(len(chars))
    ax.bar(x_pos, delta, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axhline(0,       color='black', linewidth=0.8)
    ax.axhline(mean_d, color='navy',  linewidth=1.8, linestyle='--',
               label=f'Mean = {mean_d:+.1f}%')
    ax.set_xticks(x_pos); ax.set_xticklabels(chars, fontsize=11)
    ax.set_ylabel('Fidelity improvement (%)\nAI Adaptive − Baseline', fontsize=9)
    ax.set_title('(F) Per-Character Improvement — AI Adaptive over Baseline  (Holdout Set n=30)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Stroke2Font — Generalization on 30 Held-Out {FONT_LABEL} Characters (Real Stroke Data)',
                 fontsize=13, fontweight='bold', y=0.99)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print(f"\n  [OK] Figure saved → {OUT_PNG}")

    # Also save SVG (vector) for lossless Word 2016+ / PowerPoint insertion
    out_svg = OUT_PNG.replace('.png', '.svg')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f"  [OK] Vector figure saved → {out_svg}")
    return fig


# ============================================================================
# 6.  MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STROKE2FONT  —  HOLDOUT GENERALIZATION EXPERIMENT")
    print(f"Data source: {CHAR30_FILE.split('/')[-1]}  [{FONT_LABEL}]")
    print("="*70)

    # 1. Parse data
    loader = load_char30xk if CHAR30_FILE == CHAR30_XK else load_char30ht
    print(f"\n[1] Loading {CHAR30_FILE}  [{FONT_LABEL}] ...")
    dataset = loader(CHAR30_FILE)
    print(f"    Characters loaded : {len(dataset)}")
    total_strokes = sum(e['metadata']['totalStrokes'] for e in dataset)
    total_points  = sum(e['metadata']['totalPoints']  for e in dataset)
    print(f"    Total strokes     : {total_strokes}")
    print(f"    Total feature pts : {total_points}")
    print(f"    Avg strokes/char  : {total_strokes/len(dataset):.2f}")
    print(f"    Avg pts/char      : {total_points/len(dataset):.2f}")

    # Complexity summary
    complexities = [
        e['metadata']['totalStrokes'] * len(set(s['strokeType'] for s in e['strokes']))
        for e in dataset
    ]
    print(f"    Complexity range  : {min(complexities):.0f} – {max(complexities):.0f}")

    # Save JSON
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"    [OK] Dataset saved → {OUT_JSON}")

    # 2. Evaluate
    print("\n[2] Running evaluation ...")
    results = run_evaluation(dataset)

    # 3. Summarize
    summary = summarize(results)
    strat   = stratified(results)

    # 4. Tables
    print("\n[3] Paper tables ...")
    print_tables(summary, strat, results)

    # 5. Figure
    print("[4] Generating figure ...")
    fig = plot_results(results, summary, strat)
    plt.close('all')

    print("="*70)
    print("DONE")
    print("="*70)
    return results, summary, strat


if __name__ == "__main__":
    results, summary, strat = main()
