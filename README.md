# Stroke2Font: A Hierarchical Vector Algorithm with AI-Driven Optimization for Chinese Font Generation

This repository contains the dataset, experiment code, and results for the paper:

> **Stroke2Font: A Hierarchical Vector Algorithm with AI-Driven Optimization for Chinese Font Generation**
>
> **Paper**: [https://www.mdpi.com/1999-4893/19/3/231](https://www.mdpi.com/1999-4893/19/3/231)

## Overview

Stroke2Font proposes a hierarchical stroke-element vector representation for Chinese characters combined with AI-driven trajectory optimization. The system uses:

- **Baseline**: Linear interpolation with proportional resampling
- **AI Optimized**: Cubic spline interpolation + Gaussian smoothing
- **AI Adaptive**: Per-complexity-tier parameter selection (σ and num_points)
- **DQN (Deep Q-Network)**: Reinforcement learning agent that automatically selects rotation angle (Δα) and smoothing parameter (σ) for each stroke
- **GA (Genetic Algorithm)**: Style exploration through evolutionary optimization of per-stroke parameters

## Results

| Method | Fidelity (%) | Smoothness | Time (ms) |
|--------|-------------|-----------|-----------|
| Baseline | 61.3 ± 6.5 | 0.1756 | 1.45 |
| AI Optimized | 61.2 ± 6.4 | 0.3410 | 3.81 |
| AI Adaptive | **65.2 ± 5.7** | 9.4391 | 3.67 |
| DQN (Ours) | 61.0 ± 6.3 | 0.3915 | 7.18 |

**Generalization (Holdout Test Set, n=30):**

| Method | Train | Test | Gap |
|--------|-------|------|-----|
| Baseline | 61.3% | 65.1% | 3.8% |
| AI Adaptive | 65.2% | 67.8% | 2.6% |
| DQN | 61.0% | 64.9% | 3.9% |

## Project Structure

```
├── data/
│   └── paper-1-180xk.json          # Dataset: 180 characters (150 train + 30 test)
├── src/
│   ├── stroke2font_rl_experiment.py # Main experiment: DQN training + GA + evaluation
│   ├── dqn_inference.py             # Inference: apply trained model to new characters
│   ├── performance_evaluation.py    # Baseline/Optimized/Adaptive evaluation pipeline
│   ├── holdout_evaluation.py        # Generalization validation on holdout set
│   ├── regen_stroke_comparison.py   # Generate stroke comparison figure
│   ├── regen_dataset_20chars.py     # Generate dataset overview (original)
│   └── regen_dataset_20chars_optimized.py  # Generate dataset overview (optimized)
├── results/
│   ├── dqn_model.pt                 # Trained DQN model checkpoint
│   ├── results.json                 # Full evaluation results
│   ├── training_history.npz         # DQN training curves data
│   ├── fig_training_curves.png      # DQN training curves
│   ├── fig_performance_evaluation.png  # 6-panel performance evaluation
│   ├── fig_holdout_evaluation.png   # Generalization validation
│   ├── fig_ga_convergence.png       # GA convergence curves
│   └── fig_stroke_comparison.png    # Stroke trajectory comparison
├── output/                          # Inference output directory
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

`data/paper-1-180xk.json` contains 180 Chinese characters in XingKai (行楷) font:
- **Training set**: 150 characters (index 0–149)
- **Test set**: 30 characters (index 150–179)

Each character entry includes:
```json
{
  "character": "啊",
  "metadata": { "strokeCount": 3, "totalPoints": 36, ... },
  "strokes": [
    {
      "strokeType": "折转笔画",
      "coordinates": [[x1, y1], [x2, y2], ...],
      "pointCount": 12
    }
  ]
}
```

**Stroke element types**: 横向笔画, 纵向笔画, 撇捺笔画, 折转笔画, 复合笔画, 点

**Complexity scoring**: `C = N_strokes × N_unique_stroke_types`
- Simple: C < 20
- Medium: 20 ≤ C < 50
- Complex: C ≥ 50

## Quick Start

### Requirements

```bash
pip install numpy scipy matplotlib torch
```

For GPU support (recommended for training):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Run the full experiment

```bash
python src/stroke2font_rl_experiment.py
```

This will:
1. Load the 180-character dataset
2. Train a DQN agent (500 episodes)
3. Run GA style exploration
4. Evaluate all 4 methods on train/test sets
5. Generate paper tables and 5 visualization figures
6. Save results to `results/`

### Apply trained model to new characters

```bash
# Process all 180 characters with all methods
python src/dqn_inference.py --method all --visualize

# Process a custom JSON file
python src/dqn_inference.py --input your_data.json --output your_output/

# Use GPU
python src/dqn_inference.py --device cuda

# Only DQN method, first 50 characters
python src/dqn_inference.py --method dqn --limit 50
```

### Generate individual figures

```bash
# Stroke comparison (character: 啊)
python src/regen_stroke_comparison.py

# Dataset overview — original trajectories
python src/regen_dataset_20chars.py

# Dataset overview — AI Adaptive optimized
python src/regen_dataset_20chars_optimized.py
```

## DQN Architecture

```
State (10-dim): [stroke_type_onehot(6), complexity(1), density(1), length(1), budget(1)]
    ↓
FC Layer 1: 10 → 64 (ReLU)
    ↓
FC Layer 2: 64 → 32 (ReLU)
    ↓
Output: 32 → 20 (Q-values for 20 discrete actions)

Actions: 5 Δα values × 4 σ values = 20 combinations
  Δα ∈ {-10°, -5°, 0°, +5°, +10°}
  σ ∈ {0.5, 0.8, 1.0, 1.5}

Reward: R = 1.0·fidelity − 0.1·latency − 0.3·curvature_variance
```

## License

MIT License

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{li2025stroke2font,
  title={Stroke2Font: A Hierarchical Vector Algorithm with AI-Driven Optimization for Chinese Font Generation},
  author={Li, Qingsheng},
  journal={Algorithms},
  volume={19},
  number={3},
  pages={231},
  year={2026},
  publisher={MDPI},
  doi={10.3390/a19030231}
}
```
