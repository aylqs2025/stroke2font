#!/usr/bin/env python3
"""
Stroke2Font RL Experiment — Complete Pipeline
==============================================
DQN Training + GA Style Exploration + Baseline/Optimized/Adaptive Evaluation

Usage:
    python stroke2font_rl_experiment.py

Requirements:
    pip install numpy scipy matplotlib
    pip install torch --index-url https://download.pytorch.org/whl/cu121

Data:
    data/paper-1-180xk.json  (180 characters, 150 train + 30 test)

Output (results/ directory):
    dqn_model.pt              - Trained DQN model
    results.json              - All evaluation results
    training_history.npz      - Training curves data
    experiment.log            - Experiment log
    fig_training_curves.png   - DQN training curves
    fig_performance_evaluation.png - 6-panel performance figure
    fig_holdout_evaluation.png     - Generalization validation
    fig_ga_convergence.png         - GA convergence curves
    fig_stroke_comparison.png      - Stroke trajectory comparison
"""

import json
import math
import os
import sys
import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============================================================================
# Section 0: Configuration & Environment
# ============================================================================

@dataclass
class Config:
    """Centralized experiment configuration."""
    # Paths
    data_file: str = "data/paper-1-180xk.json"
    results_dir: str = "results"

    # Data split
    n_train: int = 150
    n_test: int = 30
    n_pilot: int = 50       # DQN training subset

    # Complexity tiers
    tier_simple_max: float = 20
    tier_medium_max: float = 50

    # DQN hyperparameters
    dqn_episodes: int = 500
    dqn_lr: float = 0.001
    dqn_gamma: float = 0.95
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.1
    dqn_epsilon_decay_episodes: int = 300
    dqn_batch_size: int = 32
    dqn_replay_size: int = 10000
    dqn_target_update: int = 50
    dqn_state_dim: int = 10
    dqn_action_dim: int = 20   # 5 Δα × 4 σ
    dqn_hidden1: int = 64
    dqn_hidden2: int = 32

    # Action space
    delta_alpha_values: list = field(
        default_factory=lambda: [-10, -5, 0, 5, 10])   # degrees
    sigma_values: list = field(
        default_factory=lambda: [0.5, 0.8, 1.0, 1.5])

    # Reward weights
    reward_latency: float = -0.1
    reward_fidelity: float = 1.0
    reward_curvature: float = -0.3

    # GA hyperparameters
    ga_pop_size: int = 50
    ga_generations: int = 100
    ga_crossover_rate: float = 0.8
    ga_mutation_rate: float = 0.1
    ga_tournament_k: int = 3
    ga_elitism: int = 2
    ga_alpha_mutation_std: float = 5.0     # degrees
    ga_eta_mutation_std: float = 0.3       # sigma units
    ga_alpha_range: Tuple[float, float] = (-15.0, 15.0)
    ga_eta_range: Tuple[float, float] = (0.1, 2.0)

    # Random seed
    seed: int = 42


# 6 stroke element types used in state encoding
STROKE_TYPES = ['横向笔画', '纵向笔画', '撇捺笔画', '折转笔画', '复合笔画', '点']


def setup_environment(cfg: Config):
    """Set random seeds, configure GPU, create output directory, init logging."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    os.makedirs(cfg.results_dir, exist_ok=True)

    # Setup logging — file + console
    log_path = os.path.join(cfg.results_dir, 'experiment.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")

    # Matplotlib font config (Chinese character support)
    # Try to find a CJK font available on this system
    import matplotlib.font_manager as fm
    cjk_candidates = [
        'SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
        'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 'Source Han Sans SC', 'Source Han Sans CN',
        'Droid Sans Fallback']
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = [f for f in cjk_candidates if f in available]
    if not chosen:
        logging.warning("No CJK font found. Chinese characters may show as □. "
                        "Install: sudo apt install fonts-noto-cjk")
    plt.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    import warnings
    warnings.filterwarnings('ignore', message='Glyph .* missing from')

    return device


# ============================================================================
# Section 1: Data Loading
# ============================================================================

@dataclass
class Character:
    """Parsed character with metadata."""
    name: str
    strokes: List[Dict]
    n_strokes: int
    n_points: int
    complexity: float
    tier: str
    stroke_types: List[str]


def load_dataset(cfg: Config) -> Tuple[List[Character], List[Character], List[Character]]:
    """Load JSON dataset → train / test / pilot splits."""
    # Resolve data path (works from project root or src/ directory)
    data_path = cfg.data_file
    if not os.path.exists(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', cfg.data_file)
    if not os.path.exists(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'data', 'paper-1-180xk.json')

    logging.info(f"Loading data: {os.path.abspath(data_path)}")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Split raw data FIRST (before filtering) to preserve 150/30 boundary
    train_raw = raw_data[:cfg.n_train]
    test_raw = raw_data[cfg.n_train:cfg.n_train + cfg.n_test]

    def parse_entries(entries):
        """Parse JSON entries into Character objects, skipping bad data."""
        chars = []
        for entry in entries:
            char_name = entry['character']
            strokes = entry['strokes']
            n_strokes = len(strokes)
            n_points = sum(
                s.get('pointCount', len(s.get('coordinates', [])))
                for s in strokes)

            # Skip corrupt entry (● with 1 stroke, 1 point)
            if n_strokes <= 1 and n_points <= 1:
                logging.info(f"  Skipping bad entry: '{char_name}' "
                             f"({n_strokes} strokes, {n_points} points)")
                continue

            stroke_type_set = list(set(
                s.get('strokeType', '点') for s in strokes))
            complexity = n_strokes * len(stroke_type_set)

            if complexity < cfg.tier_simple_max:
                tier = 'Simple'
            elif complexity < cfg.tier_medium_max:
                tier = 'Medium'
            else:
                tier = 'Complex'

            chars.append(Character(
                name=char_name, strokes=strokes,
                n_strokes=n_strokes, n_points=n_points,
                complexity=complexity, tier=tier,
                stroke_types=stroke_type_set))
        return chars

    train_chars = parse_entries(train_raw)
    test_chars = parse_entries(test_raw)

    # Stratified pilot set (50 chars from training)
    simple  = [c for c in train_chars if c.tier == 'Simple']
    medium  = [c for c in train_chars if c.tier == 'Medium']
    complex_ = [c for c in train_chars if c.tier == 'Complex']

    total = len(simple) + len(medium) + len(complex_)
    n_s = max(1, round(cfg.n_pilot * len(simple) / max(total, 1)))
    n_m = max(1, round(cfg.n_pilot * len(medium) / max(total, 1)))
    n_c = max(1, cfg.n_pilot - n_s - n_m)

    random.shuffle(simple)
    random.shuffle(medium)
    random.shuffle(complex_)
    pilot_chars = simple[:n_s] + medium[:n_m] + complex_[:n_c]
    random.shuffle(pilot_chars)

    logging.info(f"Dataset: {len(train_chars) + len(test_chars)} valid characters "
                 f"(from {len(raw_data)} entries)")
    logging.info(f"  Train: {len(train_chars)}, Test: {len(test_chars)}")
    logging.info(f"  Pilot: {len(pilot_chars)} "
                 f"(S={min(n_s, len(simple))}, "
                 f"M={min(n_m, len(medium))}, "
                 f"C={min(n_c, len(complex_))})")
    logging.info(f"  Avg strokes/char: "
                 f"{np.mean([c.n_strokes for c in train_chars]):.2f}")
    logging.info(f"  Avg points/char: "
                 f"{np.mean([c.n_points for c in train_chars]):.1f}")

    return train_chars, test_chars, pilot_chars


# ============================================================================
# Section 2: Trajectory Optimization Pipeline
# ============================================================================

def extract_coords(stroke: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract x, y arrays from stroke coordinate data."""
    coords = stroke.get('coordinates', [])
    x = np.array([float(p[0]) for p in coords if len(p) >= 2])
    y = np.array([float(p[1]) for p in coords if len(p) >= 2])
    return x, y


def rotate_coords(x: np.ndarray, y: np.ndarray,
                   delta_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate coordinate offsets by delta_deg around centroid."""
    if len(x) < 2 or abs(delta_deg) < 1e-6:
        return x.copy(), y.copy()
    cx, cy = np.mean(x), np.mean(y)
    rad = np.radians(delta_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx, dy = x - cx, y - cy
    x_rot = cx + dx * cos_a - dy * sin_a
    y_rot = cy + dx * sin_a + dy * cos_a
    return x_rot, y_rot


def baseline_interpolation(x: np.ndarray,
                            y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline: linear interpolation + proportional resampling."""
    if len(x) < 2:
        return x, y
    t = np.linspace(0, 1, len(x))
    n_pts = max(len(x), min(50, len(x) * 3))
    t_new = np.linspace(0, 1, n_pts)
    return interp1d(t, x)(t_new), interp1d(t, y)(t_new)


def ai_optimized_interpolation(x: np.ndarray, y: np.ndarray,
                                sigma: float = 1.0,
                                num_points: int = 50
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """AI Optimized: cubic spline + Gaussian smoothing."""
    if len(x) < 2:
        return x, y
    t = np.linspace(0, 1, len(x))
    # Proportional resampling (capped by num_points)
    n_pts = max(len(x), min(num_points, len(x) * 3))
    t_new = np.linspace(0, 1, n_pts)
    kind = 'cubic' if len(x) >= 4 else 'linear'
    x_new = interp1d(t, x, kind=kind)(t_new)
    y_new = interp1d(t, y, kind=kind)(t_new)
    if sigma > 0:
        x_new = gaussian_filter1d(x_new, sigma=sigma, mode='nearest')
        y_new = gaussian_filter1d(y_new, sigma=sigma, mode='nearest')
    return x_new, y_new


def ai_adaptive_optimization(x: np.ndarray, y: np.ndarray,
                              complexity: float
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """AI Adaptive: per-complexity-tier sigma AND num_points selection.

    Matches the original performance_evaluation.py behavior.
    """
    if len(x) < 2:
        return x, y
    if complexity < 20:
        num_points = len(x) * 2
        sigma = 0.5
    elif complexity < 50:
        num_points = len(x) * 2
        sigma = 0.8
    else:
        num_points = len(x) * 3
        sigma = 1.0
    return ai_optimized_interpolation(x, y, sigma, num_points)


def dqn_optimized(x: np.ndarray, y: np.ndarray,
                   delta_alpha: float,
                   sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """DQN Optimized: rotation + cubic spline + DQN-selected σ."""
    if len(x) < 2:
        return x, y
    x_rot, y_rot = rotate_coords(x, y, delta_alpha)
    return ai_optimized_interpolation(x_rot, y_rot, sigma)


# ============================================================================
# Section 3: Evaluation Metrics
# ============================================================================

def hausdorff_fidelity(x1, y1, x2, y2) -> float:
    """Trajectory fidelity: (1 - d_H / max_span) × 100%."""
    if len(x1) < 2 or len(x2) < 2:
        return 0.0
    P = np.column_stack([x1, y1])
    Q = np.column_stack([x2, y2])
    d = max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])
    span = max(float(np.ptp(x1)), float(np.ptp(y1)), 1.0)
    return max(0.0, 100.0 * (1.0 - d / span))


def curvature_array(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute curvature κ along trajectory."""
    if len(x) < 3:
        return np.array([0.0])
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx**2 + dy**2)**1.5 + 1e-10
    return num / den


def curvature_variance(x: np.ndarray, y: np.ndarray) -> float:
    """Curvature variance Var(κ) — lower = smoother."""
    return float(np.var(curvature_array(x, y)))


def normalized_smoothness(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized mean curvature (matches performance_evaluation.py)."""
    if len(x) < 3:
        return 0.0
    curv = curvature_array(x, y)
    length = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))) + 1e-10
    return float(np.mean(curv) * len(x) / length)


# ============================================================================
# Section 4: DQN Module (PyTorch)
# ============================================================================

class QNetwork(nn.Module):
    """Q-Network: 10 → 64(ReLU) → 32(ReLU) → 20."""

    def __init__(self, state_dim=10, action_dim=20, hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with ε-greedy exploration and target network."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.q_net = QNetwork(
            cfg.dqn_state_dim, cfg.dqn_action_dim,
            cfg.dqn_hidden1, cfg.dqn_hidden2).to(device)
        self.target_net = QNetwork(
            cfg.dqn_state_dim, cfg.dqn_action_dim,
            cfg.dqn_hidden1, cfg.dqn_hidden2).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.dqn_lr)
        self.buffer = ReplayBuffer(cfg.dqn_replay_size)
        self.epsilon = cfg.dqn_epsilon_start

        # Build action lookup: index → (Δα, σ)
        self.actions = []
        for da in cfg.delta_alpha_values:
            for s in cfg.sigma_values:
                self.actions.append((da, s))

    def build_state(self, stroke: Dict, complexity: float,
                    n_strokes: int, stroke_idx: int) -> np.ndarray:
        """Build 10-dimensional state vector.

        Layout: [stroke_type_onehot(6), complexity(1), style(2), budget(1)]
        """
        stroke_type = stroke.get('strokeType', '点')
        type_idx = (STROKE_TYPES.index(stroke_type)
                    if stroke_type in STROKE_TYPES else 5)
        one_hot = [0.0] * 6
        one_hot[type_idx] = 1.0

        # Normalized complexity score
        complexity_norm = min(complexity / 100.0, 1.0)

        # Style features: point density & stroke length
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

        # Budget: fraction of remaining strokes
        budget = 1.0 - stroke_idx / max(n_strokes, 1)

        return np.array(
            one_hot + [complexity_norm, density, length_norm, budget],
            dtype=np.float32)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.cfg.dqn_action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def decode_action(self, action: int) -> Tuple[float, float]:
        """Decode action index → (Δα degrees, σ)."""
        return self.actions[action]

    def update(self) -> float:
        """Sample minibatch and perform one gradient step. Returns loss."""
        if len(self.buffer) < self.cfg.dqn_batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.cfg.dqn_batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q(s, a)
        q_values = self.q_net(states_t).gather(
            1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.cfg.dqn_gamma * next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(q_values, target)  # Huber loss — robust to outliers

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent loss explosion
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        """Hard update: copy Q-network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self, episode: int):
        """Linear ε decay from start to end over decay_episodes."""
        if episode <= self.cfg.dqn_epsilon_decay_episodes:
            self.epsilon = (self.cfg.dqn_epsilon_start -
                (episode / self.cfg.dqn_epsilon_decay_episodes) *
                (self.cfg.dqn_epsilon_start - self.cfg.dqn_epsilon_end))
        else:
            self.epsilon = self.cfg.dqn_epsilon_end


# ============================================================================
# Section 5: GA Module
# ============================================================================

class GAOptimizer:
    """Genetic Algorithm for style parameter exploration.

    Genome: [α₁, η₁, α₂, η₂, ..., αₙ, ηₙ]
    where αᵢ = rotation angle, ηᵢ = smoothing σ for stroke i.
    """

    def __init__(self, cfg: Config, character: Character):
        self.cfg = cfg
        self.character = character
        self.genome_size = character.n_strokes * 2
        self.population = self._init_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def _init_population(self) -> np.ndarray:
        """Random initialization within parameter ranges."""
        pop = np.zeros((self.cfg.ga_pop_size, self.genome_size))
        for i in range(self.cfg.ga_pop_size):
            for j in range(0, self.genome_size, 2):
                pop[i, j] = np.random.uniform(*self.cfg.ga_alpha_range)
                pop[i, j+1] = np.random.uniform(*self.cfg.ga_eta_range)
        return pop

    def fitness(self, genome: np.ndarray) -> float:
        """Average fidelity across all strokes with genome parameters."""
        fidelities = []
        for i, stroke in enumerate(self.character.strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2:
                continue
            alpha = genome[i * 2]
            eta = genome[i * 2 + 1]
            x_opt, y_opt = dqn_optimized(x, y, alpha, eta)
            fidelities.append(hausdorff_fidelity(x, y, x_opt, y_opt))
        return float(np.mean(fidelities)) if fidelities else 0.0

    def tournament_select(self, fitnesses: np.ndarray) -> int:
        """Tournament selection with k competitors."""
        candidates = random.sample(range(len(fitnesses)),
                                   self.cfg.ga_tournament_k)
        return max(candidates, key=lambda i: fitnesses[i])

    def crossover(self, p1: np.ndarray,
                  p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        if random.random() > self.cfg.ga_crossover_rate:
            return p1.copy(), p2.copy()
        point = random.randint(1, self.genome_size - 1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Gaussian mutation with clipping."""
        g = genome.copy()
        for j in range(0, self.genome_size, 2):
            if random.random() < self.cfg.ga_mutation_rate:
                g[j] += np.random.normal(0, self.cfg.ga_alpha_mutation_std)
                g[j] = np.clip(g[j], *self.cfg.ga_alpha_range)
            if random.random() < self.cfg.ga_mutation_rate:
                g[j+1] += np.random.normal(0, self.cfg.ga_eta_mutation_std)
                g[j+1] = np.clip(g[j+1], *self.cfg.ga_eta_range)
        return g

    def evolve(self) -> Tuple[np.ndarray, float]:
        """Run GA evolution. Returns (best_genome, best_fitness)."""
        best_overall = None
        best_fitness_overall = -1
        stagnation = 0

        for gen in range(self.cfg.ga_generations):
            # Evaluate population
            fitnesses = np.array(
                [self.fitness(ind) for ind in self.population])

            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            avg_fit = np.mean(fitnesses)
            self.best_fitness_history.append(float(best_fit))
            self.avg_fitness_history.append(float(avg_fit))

            if best_fit > best_fitness_overall:
                best_fitness_overall = float(best_fit)
                best_overall = self.population[best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1

            # Early stopping if no improvement for 20 generations
            if stagnation > 20:
                break

            # Elitism: preserve top individuals
            elite_idx = np.argsort(fitnesses)[-self.cfg.ga_elitism:]
            new_pop = [self.population[i].copy() for i in elite_idx]

            # Generate offspring
            while len(new_pop) < self.cfg.ga_pop_size:
                p1 = self.tournament_select(fitnesses)
                p2 = self.tournament_select(fitnesses)
                c1, c2 = self.crossover(
                    self.population[p1], self.population[p2])
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.cfg.ga_pop_size:
                    new_pop.append(self.mutate(c2))

            self.population = np.array(new_pop[:self.cfg.ga_pop_size])

        return best_overall, best_fitness_overall


# ============================================================================
# Section 6: DQN Training Loop
# ============================================================================

def train_dqn(agent: DQNAgent, pilot_chars: List[Character],
              cfg: Config) -> Dict:
    """Train DQN agent on pilot set (500 episodes). Returns history dict."""
    logging.info(f"\n{'='*70}")
    logging.info(f"DQN TRAINING — {cfg.dqn_episodes} episodes, "
                 f"pilot set = {len(pilot_chars)} chars")
    logging.info(f"{'='*70}")

    history = {
        'episode_rewards': [],
        'episode_losses': [],
        'epsilons': [],
        'episode_fidelities': [],
    }

    t_start = time.time()

    for episode in range(1, cfg.dqn_episodes + 1):
        # Sample a random character from pilot set
        char = random.choice(pilot_chars)

        ep_reward = 0.0
        ep_loss = 0.0
        ep_fidelities = []
        n_updates = 0

        for i, stroke in enumerate(char.strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2:
                continue

            # Build state
            state = agent.build_state(
                stroke, char.complexity, char.n_strokes, i)

            # Select action (ε-greedy)
            action = agent.select_action(state)
            delta_alpha, sigma = agent.decode_action(action)

            # Execute: optimize with selected parameters
            t0 = time.perf_counter()
            x_opt, y_opt = dqn_optimized(x, y, delta_alpha, sigma)
            latency = time.perf_counter() - t0

            # Compute reward components (normalized to prevent divergence)
            fidelity = hausdorff_fidelity(x, y, x_opt, y_opt) / 100.0  # [0, 1]
            curv_var = min(curvature_variance(x_opt, y_opt), 1.0)      # clip to [0, 1]

            reward = (cfg.reward_fidelity * fidelity +
                      cfg.reward_latency * latency +
                      cfg.reward_curvature * curv_var)
            reward = float(np.clip(reward, -1.0, 1.0))  # clip total reward

            # Next state
            done = (i >= char.n_strokes - 1)
            if not done and i + 1 < len(char.strokes):
                next_state = agent.build_state(
                    char.strokes[i + 1], char.complexity,
                    char.n_strokes, i + 1)
            else:
                next_state = np.zeros(cfg.dqn_state_dim, dtype=np.float32)
                done = True

            # Store transition
            agent.buffer.push(state, action, reward, next_state, done)

            # Update Q-network
            loss = agent.update()

            ep_reward += reward
            ep_loss += loss
            ep_fidelities.append(fidelity * 100)
            n_updates += 1

        # Decay exploration rate
        agent.decay_epsilon(episode)

        # Update target network periodically
        if episode % cfg.dqn_target_update == 0:
            agent.update_target()

        # Record episode stats
        avg_reward = ep_reward / max(n_updates, 1)
        avg_loss = ep_loss / max(n_updates, 1)
        avg_fid = float(np.mean(ep_fidelities)) if ep_fidelities else 0.0

        history['episode_rewards'].append(avg_reward)
        history['episode_losses'].append(avg_loss)
        history['epsilons'].append(agent.epsilon)
        history['episode_fidelities'].append(avg_fid)

        # Progress log every 50 episodes
        if episode % 50 == 0 or episode == 1:
            elapsed = time.time() - t_start
            logging.info(
                f"  Ep {episode:4d}/{cfg.dqn_episodes} | "
                f"ε={agent.epsilon:.3f} | "
                f"R={avg_reward:+.4f} | "
                f"loss={avg_loss:.4f} | "
                f"fid={avg_fid:.1f}% | "
                f"buf={len(agent.buffer):5d} | "
                f"{elapsed:.0f}s")

    total_time = time.time() - t_start
    logging.info(f"\nDQN training complete: {total_time:.1f}s "
                 f"({total_time/60:.1f} min)")

    return history


# ============================================================================
# Section 7: Full Evaluation
# ============================================================================

def evaluate_method(chars: List[Character], method: str,
                    agent: Optional[DQNAgent] = None) -> List[Dict]:
    """Evaluate all characters with a given method."""
    results = []

    for char in chars:
        fidelities, smoothness_vals, times = [], [], []

        for i, stroke in enumerate(char.strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2:
                continue

            t0 = time.perf_counter()

            if method == 'baseline':
                x_opt, y_opt = baseline_interpolation(x, y)
            elif method == 'ai_optimized':
                x_opt, y_opt = ai_optimized_interpolation(x, y, sigma=1.0)
            elif method == 'ai_adaptive':
                x_opt, y_opt = ai_adaptive_optimization(
                    x, y, char.complexity)
            elif method == 'dqn' and agent is not None:
                state = agent.build_state(
                    stroke, char.complexity, char.n_strokes, i)
                action = agent.select_action(state, greedy=True)
                da, sig = agent.decode_action(action)
                x_opt, y_opt = dqn_optimized(x, y, da, sig)
            else:
                x_opt, y_opt = x, y

            elapsed_ms = (time.perf_counter() - t0) * 1000

            fidelities.append(hausdorff_fidelity(x, y, x_opt, y_opt))
            smoothness_vals.append(normalized_smoothness(x_opt, y_opt))
            times.append(elapsed_ms)

        results.append({
            'character': char.name,
            'stroke_count': char.n_strokes,
            'complexity': char.complexity,
            'tier': char.tier,
            'avg_fidelity': float(np.mean(fidelities)) if fidelities else 0.0,
            'std_fidelity': float(np.std(fidelities)) if fidelities else 0.0,
            'avg_smoothness': float(np.mean(smoothness_vals))
                if smoothness_vals else 0.0,
            'total_time': float(np.sum(times)),
            'avg_time': float(np.mean(times)) if times else 0.0,
        })

    return results


def run_full_evaluation(train_chars: List[Character],
                        test_chars: List[Character],
                        agent: DQNAgent, cfg: Config) -> Dict:
    """4 methods × 2 datasets → all_results dict."""
    logging.info(f"\n{'='*70}")
    logging.info("FULL EVALUATION — 4 methods × 2 datasets")
    logging.info(f"{'='*70}")

    methods = ['baseline', 'ai_optimized', 'ai_adaptive', 'dqn']
    all_results = {}

    for method in methods:
        logging.info(f"\n  [{method.upper()}]")
        ag = agent if method == 'dqn' else None

        # Training set
        train_res = evaluate_method(train_chars, method, ag)
        all_results[f'{method}_train'] = train_res
        avg_f = np.mean([r['avg_fidelity'] for r in train_res])
        avg_s = np.mean([r['avg_smoothness'] for r in train_res])
        avg_t = np.mean([r['total_time'] for r in train_res])
        logging.info(f"    Train ({len(train_res)} chars): "
                     f"fid={avg_f:.1f}%, smooth={avg_s:.4f}, "
                     f"time={avg_t:.2f}ms")

        # Test set
        test_res = evaluate_method(test_chars, method, ag)
        all_results[f'{method}_test'] = test_res
        avg_f = np.mean([r['avg_fidelity'] for r in test_res])
        logging.info(f"    Test  ({len(test_res)} chars): "
                     f"fid={avg_f:.1f}%")

    return all_results


# ============================================================================
# Section 8: Results Output (Paper Tables)
# ============================================================================

def compute_summary(results: List[Dict]) -> Dict:
    """Summary statistics from per-character results."""
    if not results:
        return {'avg_fidelity': 0, 'std_fidelity': 0,
                'avg_smoothness': 0, 'std_smoothness': 0,
                'avg_time': 0, 'std_time': 0, 'n': 0}
    fids = [r['avg_fidelity'] for r in results]
    sms = [r['avg_smoothness'] for r in results]
    times = [r['total_time'] for r in results]
    return {
        'avg_fidelity': float(np.mean(fids)),
        'std_fidelity': float(np.std(fids)),
        'avg_smoothness': float(np.mean(sms)),
        'std_smoothness': float(np.std(sms)),
        'avg_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'n': len(results),
    }


def compute_stratified(results: List[Dict]) -> Dict:
    """Per-complexity-tier statistics."""
    tiers = {'Simple': [], 'Medium': [], 'Complex': []}
    for r in results:
        tier = r.get('tier', 'Medium')
        if tier in tiers:
            tiers[tier].append(r)

    out = {}
    for tier, entries in tiers.items():
        if entries:
            out[tier] = {
                'n': len(entries),
                'avg_fidelity': float(np.mean(
                    [e['avg_fidelity'] for e in entries])),
                'std_fidelity': float(np.std(
                    [e['avg_fidelity'] for e in entries])),
                'avg_smoothness': float(np.mean(
                    [e['avg_smoothness'] for e in entries])),
            }
        else:
            out[tier] = {'n': 0, 'avg_fidelity': 0,
                         'std_fidelity': 0, 'avg_smoothness': 0}
    return out


def print_paper_tables(all_results: Dict):
    """Print paper-ready result tables."""
    W = 80
    methods = ['baseline', 'ai_optimized', 'ai_adaptive', 'dqn']
    labels = ['Baseline', 'AI Optimized', 'AI Adaptive', 'DQN (Ours)']

    # ── Table 12: Overall Performance (Training Set) ──
    print(f"\n{'='*W}")
    print("TABLE 12: Overall Performance Comparison "
          "(Training Set, n≈150)")
    print(f"{'='*W}")
    print(f"  {'Method':<20} {'Fidelity (%)':>16} "
          f"{'Smoothness':>14} {'Time (ms)':>12}")
    print(f"  {'-'*62}")

    for method, label in zip(methods, labels):
        key = f'{method}_train'
        if key in all_results:
            s = compute_summary(all_results[key])
            print(f"  {label:<20} "
                  f"{s['avg_fidelity']:>8.1f}±{s['std_fidelity']:<6.1f} "
                  f"{s['avg_smoothness']:>12.4f} "
                  f"{s['avg_time']:>10.2f}")

    # ── Table 13: Generalization ──
    print(f"\n{'='*W}")
    print("TABLE 13: Generalization Comparison")
    print(f"{'='*W}")
    print(f"  {'Split':<20} {'n':>5} {'Baseline':>12} "
          f"{'AI Adaptive':>14} {'DQN':>12}")
    print(f"  {'-'*63}")

    for split, split_label in [('train', 'Training'), ('test', 'Holdout')]:
        vals = []
        for m in ['baseline', 'ai_adaptive', 'dqn']:
            key = f'{m}_{split}'
            if key in all_results:
                s = compute_summary(all_results[key])
                vals.append(f"{s['avg_fidelity']:.1f}%±{s['std_fidelity']:.1f}")
            else:
                vals.append("—")
        n = len(all_results.get(f'baseline_{split}', []))
        print(f"  {split_label:<20} {n:>5} "
              f"{vals[0]:>12} {vals[1]:>14} {vals[2]:>12}")

    # Generalization gap
    print()
    for m in ['baseline', 'ai_adaptive', 'dqn']:
        tr = compute_summary(all_results.get(f'{m}_train', []))
        te = compute_summary(all_results.get(f'{m}_test', []))
        gap = abs(tr['avg_fidelity'] - te['avg_fidelity'])
        print(f"  {m} generalization gap: {gap:.1f}%")

    # ── Complexity-Stratified Table ──
    print(f"\n{'='*W}")
    print("TABLE: Complexity-Stratified Performance (Training Set)")
    print(f"{'='*W}")
    print(f"  {'Tier':<20} {'n':>4} {'Baseline':>10} "
          f"{'AI Adaptive':>14} {'DQN':>10} {'Δ(DQN-Base)':>12}")
    print(f"  {'-'*70}")

    strats = {}
    for m in ['baseline', 'ai_adaptive', 'dqn']:
        key = f'{m}_train'
        if key in all_results:
            strats[m] = compute_stratified(all_results[key])

    for tier in ['Simple', 'Medium', 'Complex']:
        b = strats.get('baseline', {}).get(tier, {})
        a = strats.get('ai_adaptive', {}).get(tier, {})
        d = strats.get('dqn', {}).get(tier, {})
        delta = d.get('avg_fidelity', 0) - b.get('avg_fidelity', 0)
        print(f"  {tier:<20} {b.get('n', 0):>4} "
              f"{b.get('avg_fidelity', 0):>9.1f}% "
              f"{a.get('avg_fidelity', 0):>13.1f}% "
              f"{d.get('avg_fidelity', 0):>9.1f}% "
              f"{delta:>+11.1f}%")
    print()


# ============================================================================
# Section 9: Visualization (5 figures)
# ============================================================================

def plot_training_curves(history: Dict, cfg: Config) -> str:
    """Fig 1: DQN training curves — reward, loss, epsilon, fidelity."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    episodes = range(1, len(history['episode_rewards']) + 1)
    window = 20

    # (A) Reward
    ax = axes[0, 0]
    rewards = history['episode_rewards']
    ax.plot(episodes, rewards, alpha=0.3, color='#3498DB', lw=0.5)
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(rewards)+1), ma,
                color='#2C3E50', lw=2, label=f'MA(w={window})')
    ax.axhline(-0.12, color='red', ls='--', alpha=0.5, label='Target ≈ -0.12')
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Reward')
    ax.set_title('(A) Episode Reward', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (B) Loss
    ax = axes[0, 1]
    losses = history['episode_losses']
    ax.plot(episodes, losses, alpha=0.3, color='#E74C3C', lw=0.5)
    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(losses)+1), ma,
                color='#C0392B', lw=2, label=f'MA(w={window})')
    ax.set_xlabel('Episode'); ax.set_ylabel('TD Loss')
    ax.set_title('(B) Training Loss', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (C) Epsilon
    ax = axes[1, 0]
    ax.plot(episodes, history['epsilons'], color='#27AE60', lw=2)
    ax.axvline(cfg.dqn_epsilon_decay_episodes, color='gray', ls='--',
               alpha=0.5, label=f'Decay end (ep={cfg.dqn_epsilon_decay_episodes})')
    ax.set_xlabel('Episode'); ax.set_ylabel('ε')
    ax.set_title('(C) Exploration Rate', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (D) Fidelity
    ax = axes[1, 1]
    fids = history['episode_fidelities']
    ax.plot(episodes, fids, alpha=0.3, color='#9B59B6', lw=0.5)
    if len(fids) >= window:
        ma = np.convolve(fids, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(fids)+1), ma,
                color='#8E44AD', lw=2, label=f'MA(w={window})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Fidelity (%)')
    ax.set_title('(D) Episode Fidelity', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Stroke2Font — DQN Training Curves',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(cfg.results_dir, 'fig_training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  [OK] {path}")
    return path


def plot_performance_evaluation(all_results: Dict, cfg: Config) -> str:
    """Fig 2: 6-panel performance evaluation."""
    methods = ['baseline', 'ai_optimized', 'ai_adaptive', 'dqn']
    labels = ['Baseline\n(Linear)', 'AI Optimized\n(Cubic)',
              'AI Adaptive\n(Tier)', 'DQN\n(Ours)']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

    summaries = {m: compute_summary(all_results.get(f'{m}_train', []))
                 for m in methods}

    fig = plt.figure(figsize=(18, 12))

    # (A) Fidelity bars
    ax = fig.add_subplot(2, 3, 1)
    means = [summaries[m]['avg_fidelity'] for m in methods]
    stds = [summaries[m]['std_fidelity'] for m in methods]
    bars = ax.bar(range(4), means, yerr=stds, color=colors,
                  capsize=5, alpha=0.85, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Trajectory Fidelity (%)')
    ax.set_title('(A) Fidelity Comparison', fontweight='bold')
    ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    # (B) Smoothness bars
    ax = fig.add_subplot(2, 3, 2)
    sm = [summaries[m]['avg_smoothness'] for m in methods]
    ss = [summaries[m]['std_smoothness'] for m in methods]
    bars = ax.bar(range(4), sm, yerr=ss, color=colors,
                  capsize=5, alpha=0.85, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Curvature (Normalized)')
    ax.set_title('(B) Smoothness Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, sm):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    # (C) Time bars
    ax = fig.add_subplot(2, 3, 3)
    tm = [summaries[m]['avg_time'] for m in methods]
    ts = [summaries[m]['std_time'] for m in methods]
    bars = ax.bar(range(4), tm, yerr=ts, color=colors,
                  capsize=5, alpha=0.85, edgecolor='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Generation Time (ms)')
    ax.set_title('(C) Computational Efficiency', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, tm):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    # (D) Complexity scatter
    ax = fig.add_subplot(2, 3, 4)
    for m, c, mk, lab in [('baseline', '#E74C3C', 'o', 'Baseline'),
                           ('ai_adaptive', '#2ECC71', '^', 'AI Adaptive'),
                           ('dqn', '#9B59B6', 's', 'DQN')]:
        key = f'{m}_train'
        if key in all_results:
            cx = [r['complexity'] for r in all_results[key]]
            fy = [r['avg_fidelity'] for r in all_results[key]]
            ax.scatter(cx, fy, c=c, alpha=0.6, s=40, edgecolors='black',
                       lw=0.3, marker=mk, label=lab)
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('(D) Complexity vs Fidelity', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (E) Improvement heatmap
    ax = fig.add_subplot(2, 3, 5)
    base_train = all_results.get('baseline_train', [])
    if base_train:
        chars = [r['character'] for r in base_train]
        imp_matrix = []
        for m in ['ai_adaptive', 'dqn']:
            key = f'{m}_train'
            if key in all_results:
                imps = [all_results[key][i]['avg_fidelity'] -
                        base_train[i]['avg_fidelity']
                        for i in range(len(base_train))]
                imp_matrix.append(imps)
        if imp_matrix:
            n = len(chars)
            step = max(1, n // 30)
            idx = list(range(0, n, step))
            sub_chars = [chars[i] for i in idx]
            sub_matrix = np.array([[row[i] for i in idx]
                                   for row in imp_matrix])
            im = ax.imshow(sub_matrix, cmap='RdYlGn', aspect='auto',
                           vmin=-10, vmax=15)
            ax.set_xticks(range(len(sub_chars)))
            ax.set_xticklabels(sub_chars, fontsize=7)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['AI Adaptive', 'DQN'], fontsize=9)
            ax.set_title('(E) Fidelity Δ vs Baseline (%)', fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

    # (F) Summary table
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    tdata = [['Metric'] + [l.replace('\n', ' ') for l in labels]]
    tdata.append(['Fidelity (%)'] + [
        f"{summaries[m]['avg_fidelity']:.1f}±{summaries[m]['std_fidelity']:.1f}"
        for m in methods])
    tdata.append(['Smoothness'] + [
        f"{summaries[m]['avg_smoothness']:.4f}" for m in methods])
    tdata.append(['Time (ms)'] + [
        f"{summaries[m]['avg_time']:.2f}" for m in methods])
    table = ax.table(cellText=tdata, loc='center', cellLoc='center',
                     colWidths=[0.22, 0.195, 0.195, 0.195, 0.195])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    for j in range(5):
        table[(0, j)].set_facecolor('#34495E')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('(F) Summary', fontweight='bold', pad=20)

    plt.suptitle('Stroke2Font — Performance Evaluation (Training Set)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(cfg.results_dir, 'fig_performance_evaluation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  [OK] {path}")
    return path


def plot_holdout_evaluation(all_results: Dict, cfg: Config) -> str:
    """Fig 3: Train vs test generalization."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = ['baseline', 'ai_adaptive', 'dqn']
    labels = ['Baseline', 'AI Adaptive', 'DQN (Ours)']
    c_tr = ['#E74C3C', '#2ECC71', '#9B59B6']
    c_te = ['#C0392B', '#27AE60', '#8E44AD']

    # (A) Fidelity comparison
    ax = axes[0]
    tr_f = [compute_summary(
        all_results.get(f'{m}_train', []))['avg_fidelity'] for m in methods]
    tr_s = [compute_summary(
        all_results.get(f'{m}_train', []))['std_fidelity'] for m in methods]
    te_f = [compute_summary(
        all_results.get(f'{m}_test', []))['avg_fidelity'] for m in methods]
    te_s = [compute_summary(
        all_results.get(f'{m}_test', []))['std_fidelity'] for m in methods]

    x = np.arange(3); w = 0.35
    ax.bar(x-w/2, tr_f, w, yerr=tr_s, color=c_tr, alpha=0.75,
           capsize=4, edgecolor='black', label='Train (n≈150)')
    ax.bar(x+w/2, te_f, w, yerr=te_s, color=c_te, alpha=0.9,
           capsize=4, edgecolor='black', hatch='//', label='Test (n=30)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Fidelity (%)')
    ax.set_title('(A) Train vs Test Fidelity', fontweight='bold')
    ax.set_ylim(0, 85); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    for xp, yp in zip(np.concatenate([x-w/2, x+w/2]),
                       np.concatenate([tr_f, te_f])):
        ax.text(xp, yp+0.5, f'{yp:.1f}%', ha='center', va='bottom',
                fontsize=8)

    # (B) Generalization gap
    ax = axes[1]
    gaps = [abs(t - h) for t, h in zip(tr_f, te_f)]
    ax.bar(range(3), gaps, color=['#E67E22']*3, alpha=0.85, edgecolor='black')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Gap (%)')
    ax.set_title('(B) Generalization Gap (lower=better)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, g in enumerate(gaps):
        ax.text(i, g+0.1, f'{g:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # (C) Per-character test improvement
    ax = axes[2]
    tb = all_results.get('baseline_test', [])
    td = all_results.get('dqn_test', [])
    if tb and td:
        chars = [r['character'] for r in tb]
        delta = [td[i]['avg_fidelity'] - tb[i]['avg_fidelity']
                 for i in range(len(tb))]
        bar_c = ['#27AE60' if v >= 0 else '#C0392B' for v in delta]
        ax.bar(range(len(chars)), delta, color=bar_c, alpha=0.85,
               edgecolor='black', lw=0.3)
        ax.axhline(0, color='black', lw=0.8)
        md = float(np.mean(delta))
        ax.axhline(md, color='navy', lw=1.5, ls='--',
                   label=f'Mean = {md:+.1f}%')
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(chars, fontsize=9)
        ax.set_ylabel('DQN − Baseline (%)')
        ax.set_title('(C) Per-Char DQN Improvement (Test)', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Stroke2Font — Generalization Validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(cfg.results_dir, 'fig_holdout_evaluation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  [OK] {path}")
    return path


def plot_ga_convergence(ga_results: List[Dict], cfg: Config) -> str:
    """Fig 4: GA convergence curves for sample characters."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Best fitness
    ax = axes[0]
    for res in ga_results[:5]:
        ax.plot(res['best_history'], label=res['character'], alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (Fidelity %)')
    ax.set_title('(A) GA Convergence — Best Fitness', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (B) Average fitness
    ax = axes[1]
    for res in ga_results[:5]:
        ax.plot(res['avg_history'], label=res['character'], alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Avg Population Fitness (%)')
    ax.set_title('(B) GA Convergence — Population Avg', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Stroke2Font — GA Style Exploration',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(cfg.results_dir, 'fig_ga_convergence.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  [OK] {path}")
    return path


def plot_stroke_comparison(train_chars: List[Character],
                           agent: DQNAgent, cfg: Config) -> str:
    """Fig 5: Original vs optimized stroke trajectories."""
    # Pick a medium-complexity character with ≥4 strokes
    char = None
    for c in train_chars:
        if c.tier == 'Medium' and c.n_strokes >= 4:
            char = c
            break
    if char is None:
        char = train_chars[0]

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
        for i, stroke in enumerate(char.strokes):
            x, y = extract_coords(stroke)
            if len(x) < 2:
                continue
            if method is None:
                xp, yp = x, y
            elif method == 'baseline':
                xp, yp = baseline_interpolation(x, y)
            elif method == 'ai_adaptive':
                xp, yp = ai_adaptive_optimization(x, y, char.complexity)
            elif method == 'dqn':
                state = agent.build_state(
                    stroke, char.complexity, char.n_strokes, i)
                action = agent.select_action(state, greedy=True)
                da, sig = agent.decode_action(action)
                xp, yp = dqn_optimized(x, y, da, sig)
            else:
                xp, yp = x, y

            color = plt.cm.viridis(i / max(1, char.n_strokes - 1))
            ax.plot(xp, -np.array(yp), color=color, lw=2, alpha=0.8)

        ax.set_aspect('equal')
        ax.set_title(f'{name}\n{char.name}', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom row: single stroke detail
    stroke_idx = min(2, len(char.strokes) - 1)
    stroke = char.strokes[stroke_idx]
    x_orig, y_orig = extract_coords(stroke)

    if len(x_orig) >= 2:
        x_base, y_base = baseline_interpolation(x_orig, y_orig)
        x_adap, y_adap = ai_adaptive_optimization(
            x_orig, y_orig, char.complexity)
        state = agent.build_state(
            stroke, char.complexity, char.n_strokes, stroke_idx)
        action = agent.select_action(state, greedy=True)
        da, sig = agent.decode_action(action)
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
            fid = (hausdorff_fidelity(x_orig, y_orig, xp, yp)
                   if col > 0 else 100.0)
            ax.set_title(f'{name}\nκ_var={cv:.4f}, fid={fid:.1f}%',
                         fontsize=9)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)

    plt.suptitle(
        f'Stroke Comparison — "{char.name}" '
        f'({char.n_strokes} strokes, C={char.complexity:.0f})',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(cfg.results_dir, 'fig_stroke_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  [OK] {path}")
    return path


# ============================================================================
# Section 10: Save & Main Entry
# ============================================================================

def save_results(all_results: Dict, history: Dict,
                 agent: DQNAgent, ga_results: List[Dict],
                 cfg: Config):
    """Persist all experiment outputs."""

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save DQN model checkpoint
    model_path = os.path.join(cfg.results_dir, 'dqn_model.pt')
    torch.save({
        'q_net_state_dict': agent.q_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'config': {
            'state_dim': cfg.dqn_state_dim,
            'action_dim': cfg.dqn_action_dim,
            'hidden1': cfg.dqn_hidden1,
            'hidden2': cfg.dqn_hidden2,
        }
    }, model_path)
    logging.info(f"  [OK] Model: {model_path}")

    # Save training history
    hist_path = os.path.join(cfg.results_dir, 'training_history.npz')
    np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
    logging.info(f"  [OK] History: {hist_path}")

    # Save results JSON
    results_json = {}
    for key, val in all_results.items():
        if isinstance(val, list):
            results_json[key] = [
                {k: to_serializable(v) for k, v in r.items()}
                for r in val]

    # Summaries
    methods = ['baseline', 'ai_optimized', 'ai_adaptive', 'dqn']
    for split in ['train', 'test']:
        for m in methods:
            k = f'{m}_{split}'
            if k in all_results:
                results_json[f'{k}_summary'] = compute_summary(
                    all_results[k])

    # GA summary
    results_json['ga_summary'] = [
        {'character': r['character'],
         'best_fitness': to_serializable(r['best_fitness']),
         'generations': r['generations']}
        for r in ga_results]

    json_path = os.path.join(cfg.results_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2,
                  default=to_serializable)
    logging.info(f"  [OK] Results: {json_path}")


def main():
    """Main entry — orchestrates the full Stroke2Font RL experiment."""
    cfg = Config()

    # ── Step 0: Environment ───────────────────────────────
    print("=" * 70)
    print("STROKE2FONT — RL EXPERIMENT")
    print("DQN Training + GA Exploration + Full Evaluation")
    print("=" * 70)

    device = setup_environment(cfg)

    # ── Step 1: Load data ─────────────────────────────────
    logging.info("\n[Step 1/8] Loading dataset...")
    train_chars, test_chars, pilot_chars = load_dataset(cfg)

    # ── Step 2: Initialize DQN ────────────────────────────
    logging.info("\n[Step 2/8] Initializing DQN agent...")
    agent = DQNAgent(cfg, device)
    n_params = sum(p.numel() for p in agent.q_net.parameters())
    logging.info(f"  Q-Network: {cfg.dqn_state_dim} → "
                 f"{cfg.dqn_hidden1} → {cfg.dqn_hidden2} → "
                 f"{cfg.dqn_action_dim}  ({n_params} params)")
    logging.info(f"  Actions: Δα={cfg.delta_alpha_values} × "
                 f"σ={cfg.sigma_values}")

    # ── Step 3: Train DQN ─────────────────────────────────
    logging.info("\n[Step 3/8] Training DQN...")
    history = train_dqn(agent, pilot_chars, cfg)

    # ── Step 4: GA exploration ────────────────────────────
    logging.info("\n[Step 4/8] Running GA style exploration...")
    ga_sample = random.sample(train_chars, min(5, len(train_chars)))
    ga_results = []
    for char in ga_sample:
        logging.info(f"  GA: '{char.name}' ({char.n_strokes} strokes)...")
        ga = GAOptimizer(cfg, char)
        best_genome, best_fitness = ga.evolve()
        ga_results.append({
            'character': char.name,
            'best_fitness': best_fitness,
            'best_genome': best_genome.tolist() if best_genome is not None
                else [],
            'best_history': ga.best_fitness_history,
            'avg_history': ga.avg_fitness_history,
            'generations': len(ga.best_fitness_history),
        })
        logging.info(f"    → fitness={best_fitness:.1f}% "
                     f"({len(ga.best_fitness_history)} gen)")

    # ── Step 5: Full evaluation ───────────────────────────
    logging.info("\n[Step 5/8] Running full evaluation...")
    all_results = run_full_evaluation(train_chars, test_chars, agent, cfg)

    # ── Step 6: Paper tables ──────────────────────────────
    logging.info("\n[Step 6/8] Paper tables...")
    print_paper_tables(all_results)

    # ── Step 7: Visualization ─────────────────────────────
    logging.info("\n[Step 7/8] Generating figures...")
    plot_training_curves(history, cfg)
    plot_performance_evaluation(all_results, cfg)
    plot_holdout_evaluation(all_results, cfg)
    plot_ga_convergence(ga_results, cfg)
    plot_stroke_comparison(train_chars, agent, cfg)

    # ── Step 8: Save ──────────────────────────────────────
    logging.info("\n[Step 8/8] Saving results...")
    save_results(all_results, history, agent, ga_results, cfg)

    # ── Done ──────────────────────────────────────────────
    logging.info(f"\n{'='*70}")
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"Results: {os.path.abspath(cfg.results_dir)}")
    logging.info(f"{'='*70}")


if __name__ == "__main__":
    main()
