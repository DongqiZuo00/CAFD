"""CPU-only CAFD-MPC controller: method specification Appendix A (2026-09-06).

Inputs are control-probe observations, NEVER development or test results.
Each observation has one row per family: (mean reward, full-pass, E/(1+E)).
``choose_action`` reserves a block and is idempotent until ``observe`` completes
it. Save ``state_dict`` together with training state to resume a pending block.
No model, filesystem, Slurm, or GPU operations occur in this module.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MPCConfig:
    route_intervals: int
    total_rounds: int
    family_weights: tuple[float, ...]
    seed: int = 2027
    block_rounds: int = 10
    horizon: int = 2
    exploration_probability: float = 0.1
    forgetting_factor: float = 0.95
    ridge: float = 1.0
    reward_tolerance: float = 1e-4
    error_tolerance: float = 1e-8

    def __post_init__(self):
        for name in ('route_intervals', 'total_rounds', 'block_rounds', 'horizon'):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer')
            object.__setattr__(self, name, int(value))
        weights = np.asarray(self.family_weights, dtype=np.float64)
        if (weights.ndim != 1 or not len(weights) or
                not np.isfinite(weights).all() or (weights <= 0).any() or
                not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError('family_weights must be fixed strictly positive weights summing to one')
        object.__setattr__(self, 'family_weights', tuple(float(w) for w in weights))
        for name, low, high in (
            ('exploration_probability', 0, 1), ('forgetting_factor', 0, 1),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} outside [{low}, {high}]')
        if self.forgetting_factor == 0:
            raise ValueError('forgetting_factor must be positive')
        for name in ('ridge', 'reward_tolerance', 'error_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == 'ridge' and value == 0):
                raise ValueError(f'invalid {name}')

    @property
    def delta(self) -> float:
        return min(0.5, self.route_intervals / 8.0)


def project_state(state: np.ndarray) -> np.ndarray:
    """Euclidean projection in Appendix A.2, including v <= r."""
    out = np.asarray(state, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 3 or not np.isfinite(out).all():
        raise ValueError('state must be a finite (D, 3) array')
    crossed = out[:, 1] > out[:, 0]
    midpoint = (out[crossed, 0] + out[crossed, 1]) / 2.0
    out[crossed, 0] = midpoint
    out[crossed, 1] = midpoint
    return np.clip(out, 0.0, 1.0)


def effective_increment(u: float, action: int, route_intervals: int, delta: float) -> float:
    if action not in (0, 1):
        raise ValueError('action must be hold=0 or advance=1')
    return (min(float(route_intervals), u + delta * action) - u) / delta


def build_features(state, previous_state, weights, u, previous_increment,
                   action, completed_rounds, config: MPCConfig) -> np.ndarray:
    """Return D rows of the exact ordered D+17 features in equation (15)."""
    state = np.asarray(state, dtype=np.float64)
    previous_state = np.asarray(previous_state, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    d = len(weights)
    if state.shape != (d, 3) or previous_state.shape != state.shape:
        raise ValueError('observations must have shape (number of families, 3)')
    if not np.isfinite(state).all() or not np.isfinite(previous_state).all():
        raise ValueError('missing observations cannot be encoded as features')
    zeta = float(u) / config.route_intervals
    t = float(completed_rounds) / config.total_rounds
    b = effective_increment(u, action, config.route_intervals, config.delta)
    mean = weights @ state
    scalar_columns = np.tile([zeta, t, previous_increment, b, b * zeta], (d, 1))
    return np.concatenate((np.eye(d), state, state - previous_state,
                           np.tile(mean, (d, 1)), scalar_columns, b * state), axis=1)


def weighted_ridge(features, targets, sample_weights, ridge: float) -> np.ndarray:
    """Return output-by-feature W using a linear solve, penalizing ALL features."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or weights.shape != (len(x),):
        raise ValueError('invalid ridge design shapes')
    if (ridge <= 0 or not np.isfinite(ridge) or (weights < 0).any() or
            not all(np.isfinite(a).all() for a in (x, y, weights))):
        raise ValueError('ridge inputs must be finite with positive regularization')
    lhs = x.T @ (weights[:, None] * x) + ridge * np.eye(x.shape[1])
    rhs = x.T @ (weights[:, None] * y)
    return np.linalg.solve(lhs, rhs).T


def select_forecast(candidates, reward_tolerance=1e-4, error_tolerance=1e-8):
    """Global max-reward tolerance, then min-error tolerance, then early advance."""
    if not candidates:
        raise ValueError('at least one forecast is required')
    best_reward = max(c['reward'] for c in candidates)
    tied = [c for c in candidates if c['reward'] >= best_reward - reward_tolerance]
    best_error = min(c['error'] for c in tied)
    tied = [c for c in tied if c['error'] <= best_error + error_tolerance]
    return max(tied, key=lambda c: tuple(c['actions']))


class MPCController:
    """Serializable block controller; missing observations are explicitly None.

    ``initial_state`` and ``observe`` accept None or a (D,3) array. Any missing
    entry makes that boundary incomplete. Missing KL is never replaced by zero.
    ``observe(..., rounds=n)`` with n less than the reserved rounds represents an
    external compute cap, terminates this controller, and does NOT fit a partial
    transition. Resuming inside an unfinished block instead uses state_dict.
    """

    FORMAT_VERSION = 1

    def __init__(self, config: MPCConfig, initial_state):
        self.config = config
        self.weights = np.asarray(config.family_weights, dtype=np.float64)
        self.families = len(self.weights)
        self.rng = np.random.default_rng(config.seed)
        self.initial_actions = [0, 1]
        for _ in range(2):
            pair = [0, 1]
            if int(self.rng.integers(2)):
                pair.reverse()
            self.initial_actions.extend(pair)
        self.u = min(config.delta, float(config.route_intervals))
        self.state = self._observation(initial_state)
        self.previous_state = None if self.state is None else self.state.copy()
        self.previous_increment = 0.0
        self.completed_rounds = 0
        self.completed_full_blocks = 0
        self.W = np.zeros((3, self.families + 17), dtype=np.float64)
        self.transitions: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.pending = None
        self.predictive_decisions = 0
        self.mpc_multiaction_decisions = 0
        self.randomized_decisions = 0
        self.stop_reason = None

    def _observation(self, value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self.families, 3):
            raise ValueError(f'control state must have shape ({self.families}, 3)')
        if not np.isfinite(arr).all():
            return None
        if ((arr < 0).any() or (arr > 1).any() or (arr[:, 1] > arr[:, 0]).any()):
            raise ValueError('observed state violates 0 <= v <= r <= 1 or 0 <= e <= 1')
        return arr.copy()

    @property
    def remaining_rounds(self):
        return self.config.total_rounds - self.completed_rounds

    @property
    def done(self):
        return self.remaining_rounds == 0 or self.stop_reason is not None

    def admissible_actions(self, u=None):
        coordinate = self.u if u is None else float(u)
        return [0] if coordinate >= self.config.route_intervals else [0, 1]

    def _features(self, state, previous, u, previous_increment, action, rounds):
        return build_features(state, previous, self.weights, u, previous_increment,
                              action, rounds, self.config)

    def forecast_candidates(self):
        """Pure counterfactual search: does not draw RNG or mutate state/W."""
        if self.state is None or self.previous_state is None:
            return []
        horizon = min(self.config.horizon, self.remaining_rounds // self.config.block_rounds)
        if horizon == 0:
            return []
        candidates = []

        def visit(state, previous, u, previous_increment, rounds, actions, depth):
            if depth == horizon:
                candidates.append({
                    'actions': actions, 'reward': float(self.weights @ state[:, 0]),
                    'error': float(self.weights @ state[:, 2]),
                    'terminal_state': state.tolist(), 'terminal_u': float(u),
                })
                return
            for action in self.admissible_actions(u):
                features = self._features(state, previous, u, previous_increment, action, rounds)
                next_state = project_state(state + features @ self.W.T)
                b = effective_increment(u, action, self.config.route_intervals, self.config.delta)
                next_u = min(float(self.config.route_intervals), u + self.config.delta * action)
                visit(next_state, state, next_u, b, rounds + self.config.block_rounds,
                      actions + [action], depth + 1)

        visit(self.state, self.previous_state, self.u, self.previous_increment,
              self.completed_rounds, [], 0)
        return candidates

    def choose_action(self):
        """Reserve next actual block; repeated calls return the same decision."""
        if self.pending is not None:
            return copy.deepcopy(self.pending)
        if self.done:
            raise RuntimeError('training-round budget finished or external cap reached')
        cfg = self.config
        rounds = min(cfg.block_rounds, self.remaining_rounds)
        actions = self.admissible_actions()
        candidates = []
        if rounds < cfg.block_rounds:
            action, kind = 0, 'partial_hold'
        elif self.state is None or self.previous_state is None:
            action, kind = 0, 'missing_observation_hold'
        elif len(actions) == 1:
            action, kind = 0, 'endpoint_hold'
        elif self.completed_full_blocks < 6:
            action = self.initial_actions[self.completed_full_blocks]
            kind = 'initialization'
        elif self.rng.random() < cfg.exploration_probability:
            action, kind = int(self.rng.choice(actions)), 'random_exploration'
            self.randomized_decisions += 1
        else:
            candidates = self.forecast_candidates()
            best = select_forecast(candidates, cfg.reward_tolerance, cfg.error_tolerance)
            action, kind = int(best['actions'][0]), 'mpc'
            self.predictive_decisions += 1
            self.mpc_multiaction_decisions += int(len(actions) > 1)
        features = None
        if self.state is not None and self.previous_state is not None:
            features = self._features(self.state, self.previous_state, self.u,
                                      self.previous_increment, action, self.completed_rounds)
        old_u = self.u
        b = effective_increment(old_u, action, cfg.route_intervals, cfg.delta)
        self.u = min(float(cfg.route_intervals), old_u + cfg.delta * action)
        self.pending = {
            'block_index': self.completed_full_blocks,
            'completed_rounds_before': self.completed_rounds,
            'action': int(action), 'kind': kind, 'rounds': rounds,
            'u_before': float(old_u), 'u': float(self.u), 'effective_increment': float(b),
            'features': None if features is None else features.tolist(),
            'current_observation_complete': self.state is not None,
            'preceding_observation_complete': self.previous_state is not None,
            'forecasts': candidates,
        }
        return copy.deepcopy(self.pending)

    def _refit(self):
        valid = [tr for tr in self.transitions if tr['valid']]
        if not valid:
            self.W.fill(0.0)
            return
        design, responses, weights = [], [], []
        for tr in valid:
            design.extend(tr['features'])
            responses.extend(tr['delta_state'])
            decay = self.config.forgetting_factor ** (self.completed_full_blocks - 1 - tr['block_index'])
            weights.extend((decay * self.weights).tolist())
        self.W = weighted_ridge(design, responses, weights, self.config.ridge)

    def observe(self, state, rounds=None):
        """Finish reserved block and refit after full blocks only; returns log row."""
        if self.pending is None:
            raise RuntimeError('choose_action must precede observe')
        decision = copy.deepcopy(self.pending)
        actual_rounds = decision['rounds'] if rounds is None else rounds
        if (isinstance(actual_rounds, bool) or int(actual_rounds) != actual_rounds or
                not 1 <= actual_rounds <= decision['rounds']):
            raise ValueError('actual rounds must be an integer within the reserved block')
        actual_rounds = int(actual_rounds)
        observed = self._observation(state)
        full = actual_rounds == self.config.block_rounds
        valid = full and decision['features'] is not None and observed is not None
        tr = {
            'block_index': decision['block_index'], 'valid': bool(valid),
            'features': decision['features'] if valid else None,
            'delta_state': (observed - self.state).tolist() if valid else None,
            'previous_complete': decision['preceding_observation_complete'],
            'current_complete': decision['current_observation_complete'],
            'next_complete': observed is not None,
        }
        if full:
            self.transitions.append(tr)
            self.completed_full_blocks += 1
        self.completed_rounds += actual_rounds
        self.previous_state = None if self.state is None else self.state.copy()
        self.state = observed
        self.previous_increment = decision['effective_increment']
        self.pending = None
        if actual_rounds < decision['rounds']:
            self.stop_reason = 'external_compute_cap'
        if full:
            self._refit()
        decision.update({
            'actual_rounds': actual_rounds, 'full_block': full, 'fit_transition_valid': bool(valid),
            'next_observation_complete': observed is not None,
            'next_state': None if observed is None else observed.tolist(),
            'completed_rounds_after': self.completed_rounds,
        })
        self.decisions.append(decision)
        return copy.deepcopy(decision)

    def state_dict(self):
        """JSON-safe, lossless controller/RNG/history/pending-block resume state."""
        return {
            'format_version': self.FORMAT_VERSION, 'config': asdict(self.config),
            'rng_state': copy.deepcopy(self.rng.bit_generator.state),
            'initial_actions': list(self.initial_actions), 'u': float(self.u),
            'state': None if self.state is None else self.state.tolist(),
            'previous_state': None if self.previous_state is None else self.previous_state.tolist(),
            'previous_increment': float(self.previous_increment),
            'completed_rounds': self.completed_rounds, 'completed_full_blocks': self.completed_full_blocks,
            'W': self.W.tolist(), 'transitions': copy.deepcopy(self.transitions),
            'decisions': copy.deepcopy(self.decisions), 'pending': copy.deepcopy(self.pending),
            'predictive_decisions': self.predictive_decisions,
            'mpc_multiaction_decisions': self.mpc_multiaction_decisions,
            'randomized_decisions': self.randomized_decisions, 'stop_reason': self.stop_reason,
        }

    @classmethod
    def from_state_dict(cls, saved):
        saved = copy.deepcopy(saved)
        if saved.get('format_version') != cls.FORMAT_VERSION:
            raise ValueError('unsupported MPC resume format')
        cfg = MPCConfig(**saved['config'])
        result = cls(cfg, saved['state'])
        result.previous_state = result._observation(saved['previous_state'])
        result.W = np.asarray(saved['W'], dtype=np.float64)
        if result.W.shape != (3, result.families + 17) or not np.isfinite(result.W).all():
            raise ValueError('invalid saved predictor')
        result.rng.bit_generator.state = saved['rng_state']
        for key in ('initial_actions', 'u', 'previous_increment', 'completed_rounds',
                    'completed_full_blocks', 'transitions', 'decisions', 'pending',
                    'predictive_decisions', 'mpc_multiaction_decisions',
                    'randomized_decisions', 'stop_reason'):
            setattr(result, key, saved[key])
        if not 0 < result.u <= cfg.route_intervals:
            raise ValueError('invalid saved route coordinate')
        if not 0 <= result.completed_rounds <= cfg.total_rounds:
            raise ValueError('invalid saved round count')
        return result
