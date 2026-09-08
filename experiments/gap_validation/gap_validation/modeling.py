from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - older transformers only
    AutoModelForImageTextToText = None  # type: ignore


VISION_MARKERS = ("visual", "vision_tower", "vision_model", "image_encoder", "multi_modal_projector")


def load_full_policy(model_id: str, revision: str | None, *, attn_implementation: str | None = None):
    revision_kwargs = {"revision": revision} if revision else {}
    config = AutoConfig.from_pretrained(model_id, **revision_kwargs)
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    kwargs.update(revision_kwargs)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if getattr(config, "model_type", "") in {"qwen3_5", "gemma3"}:
        if AutoModelForImageTextToText is None:
            raise RuntimeError("transformers lacks AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    return model


def enforce_text_policy_trainable(model: torch.nn.Module, audit_path: Path | None = None) -> dict[str, Any]:
    frozen: list[str] = []
    trainable: list[str] = []
    frozen_numel = 0
    trainable_numel = 0
    for name, parameter in model.named_parameters():
        is_vision = any(marker in name.lower() for marker in VISION_MARKERS)
        parameter.requires_grad_(not is_vision)
        if is_vision:
            frozen.append(name)
            frozen_numel += parameter.numel()
        else:
            trainable.append(name)
            trainable_numel += parameter.numel()
    if not trainable or trainable_numel == 0:
        raise RuntimeError("no text-policy parameters were made trainable")
    if not any("lm_head" in name or "embed_tokens" in name for name in trainable):
        raise RuntimeError("LM head or token embeddings missing from trainable set")
    audit = {
        "trainable_parameter_count": trainable_numel,
        "frozen_parameter_count": frozen_numel,
        "trainable_names": trainable,
        "frozen_names": frozen,
    }
    if audit_path:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
