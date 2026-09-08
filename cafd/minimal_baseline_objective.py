"""Training-only changes for the two isolated minimal baselines.

Numerics and exact streamed KL are imported from the validated retention runner.
The control probes continue to use the original relative target.
"""
from .kd_retention_objective import (
    RewardRouting, teacher_coefficients, route_reward_groups,
    exact_linear_forward_kl, exact_linear_token_log_probs, clipped_rl_loss,
)

CONDITIONS = ("grpo", "absolute_kd")


def absolute_teacher_coefficients(u, route_ids):
    """Convex interpolation of the two adjacent Teacher checkpoints, with no S0."""
    coefficients = teacher_coefficients(u, route_ids)
    coefficients[route_ids[0]] = coefficients.get(route_ids[0], 0.) + 1.
    return {key: value for key, value in coefficients.items() if value != 0.}


def route_training_groups(rewards, full_pass, *, condition, advantage_epsilon=1e-6):
    if condition not in CONDITIONS:
        raise ValueError("unknown minimal baseline condition")
    original = route_reward_groups(
        rewards, full_pass=full_pass, kd_retention=True,
        advantage_epsilon=advantage_epsilon,
    )
    if condition == "absolute_kd":
        return original
    distill = original.distill.new_zeros(original.distill.shape)
    return RewardRouting(distill, original.rl, ~original.rl, original.advantages)
