"""CAFD-v1: ordered local-shift distillation for Manufactoria-HAS."""

from .exact_forward_kl import completion_prediction_mask
from .relative_target import relative_target_distribution, relative_target_logits

__all__ = [
    "completion_prediction_mask",
    "relative_target_distribution",
    "relative_target_logits",
]
