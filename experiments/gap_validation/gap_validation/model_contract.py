from __future__ import annotations

import json

from transformers import AutoConfig, AutoModelForImageTextToText

from .io import ARTIFACT_ROOT, atomic_json, load_yaml


def main() -> None:
    pair = load_yaml("backbones.yaml")["pairs"]["qwen"]
    result = {}
    for role in ("teacher", "student"):
        spec = pair[role]
        config = AutoConfig.from_pretrained(spec["id"], revision=spec["revision"])
        model_class = AutoModelForImageTextToText._model_mapping[type(config)]
        result[role] = {
            "model_id": spec["id"],
            "revision": spec["revision"],
            "config_class": type(config).__name__,
            "model_type": config.model_type,
            "resolved_model_class": model_class.__name__,
        }
    atomic_json(ARTIFACT_ROOT / "model_load_contract.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
