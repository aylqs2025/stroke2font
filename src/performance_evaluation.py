"""
Performance Evaluation of AI-Driven Optimization for Chinese Font Generation
Stroke2Font - Full Model Evaluation Pipeline
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Load the data - Use 150 character dataset
data_path = r"d:\APP\python\mypaper-program\algorithms-1\data\hanzi_dataset_150_chars.json"
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

@dataclass
class StrokeMetrics:
    """Metrics for a single stroke"""
    smoothness: float      # Curvature variance (lower = smoother)
    length: float          # Total stroke length
    point_density: float   # Points per unit length
    direction_changes: int # Number of direction changes


@dataclass
class CharacterMetrics:
    """Metrics for a complete character"""
    character: str
    stroke_count: int
    complexity_score: float
    total_length: float
    avg_smoothness: float
    stroke_type_diversity: int
    generation_time: float


# ============================================================================
# 2. STROKE ANALYSIS FUNCTIONS
# ============================================================================

def extract_stroke_coords(stroke_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract coordinates from stroke data"""
    if 'coordinates' not in stroke_data:
        return np.array([]), np.array([])

    coords = stroke_data['coordinates']
    x_vals, y_vals = [], []

    for point in coords:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            if point[0] is not None and point[1] is not None:
                x_vals.append(float(point[0]))
                y_vals.append(float(point[1]))

    return np.array(x_vals), np.array(y_vals)


def calculate_stroke_length(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate total length of a stroke trajectory"""
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def calculate_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate curvature along the stroke"""
    if len(x) < 3:
        return np.array([0.0])

    # First and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5 + 1e-10  # Avoid division by zero

    return numerator / denominator


def calculate_smoothness(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate smoothness score (normalized mean curvature)"""
    curvature = calculate_curvature(x, y)
    # Normalize by trajectory length to make it comparable across different sampling rates
    length = calculate_stroke_length(x, y)
    if length > 0:
        return float(np.mean(curvature) * len(x) / length)  # Normalized smoothness
    return float(np.mean(curvature))


def count_direction_changes(x: np.ndarray, y: np.ndarray, threshold: float = 0.3) -> int:
    """Count significant direction changes in stroke"""
    if len(x) < 3:
        return 0

    dx = np.diff(x)
    dy = np.diff(y)
    angles = np.arctan2(dy, dx)
    angle_changes = np.abs(np.diff(angles))

    # Normalize to [0, pi]
    angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)

    return int(np.sum(angle_changes > threshold))


def analyze_stroke(stroke_data: dict) -> StrokeMetrics:
    """Analyze a single stroke and return metrics"""
    x, y = extract_stroke_coords(stroke_data)

    if len(x) < 2:
        return StrokeMetrics(0, 0, 0, 0)

    length = calculate_stroke_length(x, y)
    smoothness = calculate_smoothness(x, y)
    point_density = len(x) / (length + 1e-10)
    direction_changes = count_direction_changes(x, y)

    return StrokeMetrics(smoothness, length, point_density, direction_changes)


# ============================================================================
# 3. AI OPTIMIZATION SIMULATION
# ============================================================================

def baseline_interpolation(x: np.ndarray, y: np.ndarray, num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline: Simple linear interpolation (no optimization)"""
    if len(x) < 2:
        return x, y

    t = np.linspace(0, 1, len(x))
    # Proportional resampling
    num_points = max(len(x), min(num_points, len(x) * 3))
    t_new = np.linspace(0, 1, num_points)

    fx = interp1d(t, x, kind='linear')
    fy = interp1d(t, y, kind='linear')

    return fx(t_new), fy(t_new)


def ai_optimized_interpolation(x: np.ndarray, y: np.ndarray, num_points: int = 50,
                                smooth_sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """AI-Optimized: Cubic interpolation with moderate Gaussian smoothing"""
    if len(x) < 2:
        return x, y

    t = np.linspace(0, 1, len(x))
    # Use proportional resampling to avoid over-interpolation
    num_points = max(len(x), min(num_points, len(x) * 3))
    t_new = np.linspace(0, 1, num_points)

    # Cubic spline interpolation for smoother curves
    kind = 'cubic' if len(x) >= 4 else 'linear'
    fx = interp1d(t, x, kind=kind)
    fy = interp1d(t, y, kind=kind)

    x_new = fx(t_new)
    y_new = fy(t_new)

    # Apply light Gaussian smoothing only if beneficial
    if smooth_sigma > 0:
        x_smooth = gaussian_filter1d(x_new, sigma=smooth_sigma, mode='nearest')
        y_smooth = gaussian_filter1d(y_new, sigma=smooth_sigma, mode='nearest')
    else:
        x_smooth, y_smooth = x_new, y_new

    return x_smooth, y_smooth


def ai_adaptive_optimization(x: np.ndarray, y: np.ndarray,
                             complexity: float) -> Tuple[np.ndarray, np.ndarray]:
    """AI Adaptive: Adjust optimization based on stroke complexity"""
    if len(x) < 2:
        return x, y

    # Adaptive parameters based on complexity - conservative approach
    if complexity < 20:
        num_points = len(x) * 2
        smooth_sigma = 0.5
    elif complexity < 50:
        num_points = len(x) * 2
        smooth_sigma = 0.8
    else:
        num_points = len(x) * 3
        smooth_sigma = 1.0

    return ai_optimized_interpolation(x, y, num_points, smooth_sigma)


# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================

def calculate_frechet_distance(x1: np.ndarray, y1: np.ndarray,
                               x2: np.ndarray, y2: np.ndarray) -> float:
    """Calculate Fréchet distance between two trajectories"""
    if len(x1) < 2 or len(x2) < 2:
        return 0.0

    P = np.column_stack([x1, y1])
    Q = np.column_stack([x2, y2])

    # Use Hausdorff as approximation
    d1 = directed_hausdorff(P, Q)[0]
    d2 = directed_hausdorff(Q, P)[0]

    return max(d1, d2)


def calculate_trajectory_similarity(orig_x: np.ndarray, orig_y: np.ndarray,
                                    gen_x: np.ndarray, gen_y: np.ndarray) -> float:
    """Calculate similarity score (0-100, higher is better)"""
    frechet = calculate_frechet_distance(orig_x, orig_y, gen_x, gen_y)

    # Normalize to similarity score
    max_dist = max(np.ptp(orig_x), np.ptp(orig_y), 1.0)
    similarity = max(0, 100 * (1 - frechet / max_dist))

    return similarity


def evaluate_character(char_data: dict, method: str = 'ai_optimized') -> Dict:
    """Evaluate a single character with specified method"""
    char = char_data['character']
    strokes = char_data['strokes']

    start_time = time.perf_counter()

    results = {
        'character': char,
        'stroke_count': len(strokes),
        'stroke_metrics': [],
        'total_smoothness': 0,
        'total_similarity': 0,
        'stroke_types': set()
    }

    for stroke in strokes:
        x, y = extract_stroke_coords(stroke)
        stroke_type = stroke.get('strokeType', 'Unknown')
        results['stroke_types'].add(stroke_type)

        if len(x) < 2:
            continue

        # Apply optimization method
        if method == 'baseline':
            x_opt, y_opt = baseline_interpolation(x, y)
        elif method == 'ai_optimized':
            x_opt, y_opt = ai_optimized_interpolation(x, y)
        elif method == 'ai_adaptive':
            complexity = len(strokes) * len(results['stroke_types'])
            x_opt, y_opt = ai_adaptive_optimization(x, y, complexity)
        else:
            x_opt, y_opt = x, y

        # Calculate metrics
        orig_smoothness = calculate_smoothness(x, y)
        opt_smoothness = calculate_smoothness(x_opt, y_opt)
        similarity = calculate_trajectory_similarity(x, y, x_opt, y_opt)

        # Calculate improvement (positive = smoother, lower curvature)
        if orig_smoothness > 1e-10:
            improvement = (orig_smoothness - opt_smoothness) / orig_smoothness * 100
        else:
            improvement = 0

        results['stroke_metrics'].append({
            'type': stroke_type,
            'original_smoothness': orig_smoothness,
            'optimized_smoothness': opt_smoothness,
            'smoothness_improvement': improvement,
            'similarity': similarity
        })

        results['total_smoothness'] += opt_smoothness
        results['total_similarity'] += similarity

    end_time = time.perf_counter()

    n_strokes = max(len(results['stroke_metrics']), 1)
    results['avg_smoothness'] = results['total_smoothness'] / n_strokes
    results['avg_similarity'] = results['total_similarity'] / n_strokes
    results['generation_time'] = (end_time - start_time) * 1000  # ms
    results['complexity_score'] = len(strokes) * len(results['stroke_types'])
    results['stroke_types'] = list(results['stroke_types'])

    return results


# ============================================================================
# 5. FULL EVALUATION PIPELINE
# ============================================================================

def run_full_evaluation(data: List[dict]) -> Dict:
    """Run complete evaluation across all methods"""

    methods = ['baseline', 'ai_optimized', 'ai_adaptive']
    all_results = {method: [] for method in methods}

    print("\n" + "=" * 70)
    print("PERFORMANCE EVALUATION - AI-DRIVEN OPTIMIZATION")
    print("=" * 70)

    for method in methods:
        print(f"\nEvaluating method: {method.upper()}")
        print("-" * 40)

        for char_data in data:
            result = evaluate_character(char_data, method)
            all_results[method].append(result)
            print(f"  {result['character']}: "
                  f"Smoothness={result['avg_smoothness']:.4f}, "
                  f"Similarity={result['avg_similarity']:.1f}%, "
                  f"Time={result['generation_time']:.2f}ms")

    return all_results


def compute_summary_statistics(results: Dict) -> Dict:
    """Compute summary statistics for each method"""
    summary = {}

    for method, method_results in results.items():
        smoothness_values = [r['avg_smoothness'] for r in method_results]
        similarity_values = [r['avg_similarity'] for r in method_results]
        time_values = [r['generation_time'] for r in method_results]
        complexity_values = [r['complexity_score'] for r in method_results]

        # Calculate improvement percentages
        smoothness_improvements = []
        for r in method_results:
            for sm in r['stroke_metrics']:
                if sm['smoothness_improvement'] > -1000:  # Filter outliers
                    smoothness_improvements.append(sm['smoothness_improvement'])

        summary[method] = {
            'avg_smoothness': np.mean(smoothness_values),
            'std_smoothness': np.std(smoothness_values),
            'avg_similarity': np.mean(similarity_values),
            'std_similarity': np.std(similarity_values),
            'avg_time': np.mean(time_values),
            'std_time': np.std(time_values),
            'avg_improvement': np.mean(smoothness_improvements) if smoothness_improvements else 0,
            'complexity_correlation': np.corrcoef(complexity_values, smoothness_values)[0, 1]
        }

    return summary


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def create_evaluation_visualizations(results: Dict, summary: Dict):
    """Create comprehensive evaluation visualizations"""

    fig = plt.figure(figsize=(16, 12))

    methods = list(results.keys())
    method_labels = ['Baseline\n(Linear)', 'AI Optimized\n(Cubic+Smooth)', 'AI Adaptive\n(Complexity-aware)']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    # 1. Smoothness Comparison (Bar Chart) - Panel (A)
    ax1 = fig.add_subplot(2, 3, 1)
    smoothness_means = [summary[m]['avg_smoothness'] for m in methods]
    smoothness_stds = [summary[m]['std_smoothness'] for m in methods]

    bars1 = ax1.bar(range(len(methods)), smoothness_means, yerr=smoothness_stds,
                    color=colors, capsize=5, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(method_labels, fontsize=9)
    ax1.set_ylabel('Curvature Variance (Lower = Smoother)', fontsize=10)
    ax1.set_title('Stroke Smoothness Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(-0.12, 1.05, '(A)', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')

    # Add value labels
    for bar, val in zip(bars1, smoothness_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # 2. Similarity Score Comparison - Panel (B)
    ax2 = fig.add_subplot(2, 3, 2)
    similarity_means = [summary[m]['avg_similarity'] for m in methods]
    similarity_stds = [summary[m]['std_similarity'] for m in methods]

    bars2 = ax2.bar(range(len(methods)), similarity_means, yerr=similarity_stds,
                    color=colors, capsize=5, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(method_labels, fontsize=9)
    ax2.set_ylabel('Trajectory Similarity (%)', fontsize=10)
    ax2.set_title('Trajectory Fidelity Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(-0.12, 1.05, '(B)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

    for bar, val in zip(bars2, similarity_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. Generation Time Comparison - Panel (C)
    ax3 = fig.add_subplot(2, 3, 3)
    time_means = [summary[m]['avg_time'] for m in methods]
    time_stds = [summary[m]['std_time'] for m in methods]

    bars3 = ax3.bar(range(len(methods)), time_means, yerr=time_stds,
                    color=colors, capsize=5, alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(method_labels, fontsize=9)
    ax3.set_ylabel('Generation Time (ms)', fontsize=10)
    ax3.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.text(-0.12, 1.05, '(C)', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top')

    for bar, val in zip(bars3, time_means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # 4. Complexity vs Smoothness Scatter - Panel (D)
    ax4 = fig.add_subplot(2, 3, 4)
    for i, method in enumerate(methods):
        complexities = [r['complexity_score'] for r in results[method]]
        smoothness = [r['avg_smoothness'] for r in results[method]]
        ax4.scatter(complexities, smoothness, c=colors[i], label=method_labels[i].replace('\n', ' '),
                   alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

    ax4.set_xlabel('Complexity Score (Strokes x Types)', fontsize=10)
    ax4.set_ylabel('Smoothness (Curvature Variance)', fontsize=10)
    ax4.set_title('Complexity vs Smoothness', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(alpha=0.3)
    ax4.text(-0.12, 1.05, '(D)', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top')

    # 5. Per-Character Performance Heatmap - Panel (E)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.text(-0.12, 1.05, '(E)', transform=ax5.transAxes, fontsize=14, fontweight='bold', va='top')

    characters = [r['character'] for r in results['baseline']]
    improvement_matrix = []

    for method in ['ai_optimized', 'ai_adaptive']:
        improvements = []
        for i, r in enumerate(results[method]):
            baseline_smooth = results['baseline'][i]['avg_smoothness']
            method_smooth = r['avg_smoothness']
            if baseline_smooth > 0:
                imp = (baseline_smooth - method_smooth) / baseline_smooth * 100
            else:
                imp = 0
            improvements.append(imp)
        improvement_matrix.append(improvements)

    improvement_matrix = np.array(improvement_matrix)

    im = ax5.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    ax5.set_xticks(range(len(characters)))
    ax5.set_xticklabels(characters, fontsize=10)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['AI Optimized', 'AI Adaptive'], fontsize=9)
    ax5.set_title('Smoothness Improvement vs Baseline (%)', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Improvement %', fontsize=9)

    # Add text annotations
    for i in range(2):
        for j in range(len(characters)):
            val = improvement_matrix[i, j]
            color = 'white' if abs(val) > 40 else 'black'
            ax5.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    # 6. Summary Metrics Table - Panel (F)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(-0.05, 1.05, '(F)', transform=ax6.transAxes, fontsize=14, fontweight='bold', va='top')

    table_data = [
        ['Metric', 'Baseline', 'AI Optimized', 'AI Adaptive'],
        ['Avg Smoothness', f"{summary['baseline']['avg_smoothness']:.4f}",
         f"{summary['ai_optimized']['avg_smoothness']:.4f}",
         f"{summary['ai_adaptive']['avg_smoothness']:.4f}"],
        ['Avg Similarity', f"{summary['baseline']['avg_similarity']:.1f}%",
         f"{summary['ai_optimized']['avg_similarity']:.1f}%",
         f"{summary['ai_adaptive']['avg_similarity']:.1f}%"],
        ['Avg Time (ms)', f"{summary['baseline']['avg_time']:.2f}",
         f"{summary['ai_optimized']['avg_time']:.2f}",
         f"{summary['ai_adaptive']['avg_time']:.2f}"],
        ['Improvement %', '-',
         f"{summary['ai_optimized']['avg_improvement']:.1f}%",
         f"{summary['ai_adaptive']['avg_improvement']:.1f}%"],
    ]

    table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#34495E')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style first column
    for i in range(1, 5):
        table[(i, 0)].set_facecolor('#ECF0F1')
        table[(i, 0)].set_text_props(fontweight='bold')

    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Performance Evaluation: AI-Driven Optimization for Chinese Font Generation',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_stroke_comparison_visualization(results: Dict, char_idx: int = 0):
    """Visualize stroke optimization comparison for a single character"""

    char_data = data[char_idx]
    char = char_data['character']
    strokes = char_data['strokes']

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    methods = [
        ('Original', None),
        ('Baseline', 'baseline'),
        ('AI Optimized', 'ai_optimized')
    ]

    # Top row: Full character comparison
    for col, (method_name, method_key) in enumerate(methods):
        ax = axes[0, col]

        for i, stroke in enumerate(strokes):
            x, y = extract_stroke_coords(stroke)
            if len(x) < 2:
                continue

            # Apply optimization
            if method_key == 'baseline':
                x_plot, y_plot = baseline_interpolation(x, y)
            elif method_key == 'ai_optimized':
                x_plot, y_plot = ai_optimized_interpolation(x, y)
            else:
                x_plot, y_plot = x, y

            color = plt.cm.viridis(i / max(1, len(strokes) - 1))
            ax.plot(x_plot, -np.array(y_plot), color=color, linewidth=2, alpha=0.8)

        ax.set_aspect('equal')
        ax.set_title(f'{method_name}\n{char}', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.5)

    # Bottom row: Single stroke detail comparison
    stroke_idx = min(2, len(strokes) - 1)  # Select a stroke for detailed view
    stroke = strokes[stroke_idx]
    x_orig, y_orig = extract_stroke_coords(stroke)

    if len(x_orig) >= 2:
        x_base, y_base = baseline_interpolation(x_orig, y_orig)
        x_opt, y_opt = ai_optimized_interpolation(x_orig, y_orig)

        # Original
        ax = axes[1, 0]
        ax.plot(x_orig, -np.array(y_orig), 'o-', color='#E74C3C', linewidth=2, markersize=4)
        curvature = calculate_curvature(x_orig, y_orig)
        ax.set_title(f'Original Stroke\nCurvature Var: {np.var(curvature):.4f}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

        # Baseline
        ax = axes[1, 1]
        ax.plot(x_base, -np.array(y_base), '-', color='#3498DB', linewidth=2)
        curvature = calculate_curvature(x_base, y_base)
        ax.set_title(f'Baseline Interpolation\nCurvature Var: {np.var(curvature):.4f}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

        # AI Optimized
        ax = axes[1, 2]
        ax.plot(x_opt, -np.array(y_opt), '-', color='#2ECC71', linewidth=2)
        curvature = calculate_curvature(x_opt, y_opt)
        ax.set_title(f'AI Optimized\nCurvature Var: {np.var(curvature):.4f}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

    plt.suptitle(f'Stroke Optimization Comparison: Character "{char}"',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("STROKE2FONT - PERFORMANCE EVALUATION")
    print("AI-Driven Optimization for Chinese Font Generation")
    print("=" * 70)
    print(f"\nDataset: {len(data)} Chinese Characters")

    # Run full evaluation
    results = run_full_evaluation(data)

    # Compute summary statistics
    summary = compute_summary_statistics(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for method, stats in summary.items():
        print(f"\n{method.upper()}:")
        print(f"  Average Smoothness: {stats['avg_smoothness']:.4f} (+/- {stats['std_smoothness']:.4f})")
        print(f"  Average Similarity: {stats['avg_similarity']:.1f}% (+/- {stats['std_similarity']:.1f}%)")
        print(f"  Average Time: {stats['avg_time']:.2f}ms (+/- {stats['std_time']:.2f}ms)")
        print(f"  Smoothness Improvement: {stats['avg_improvement']:.1f}%")

    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    print("\n1. Creating evaluation summary visualization...")
    fig1 = create_evaluation_visualizations(results, summary)
    fig1.savefig('performance_evaluation.png', dpi=300, bbox_inches='tight')
    fig1.savefig('performance_evaluation.svg', format='svg', bbox_inches='tight')
    print("   [OK] Saved: 'performance_evaluation.png' + '.svg'")

    print("\n2. Creating stroke comparison visualization...")
    fig2 = create_stroke_comparison_visualization(results, char_idx=0)
    fig2.savefig('stroke_comparison.png', dpi=300, bbox_inches='tight')
    fig2.savefig('stroke_comparison.svg', format='svg', bbox_inches='tight')
    print("   [OK] Saved: 'stroke_comparison.png' + '.svg'")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    plt.show()

    return results, summary


if __name__ == "__main__":
    results, summary = main()
