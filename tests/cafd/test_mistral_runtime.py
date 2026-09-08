from copy import deepcopy
from pathlib import Path

import yaml

from cafd.mistral_runtime import RUN_ID, validate_config
from cafd.prompting import (
    MISTRAL_SYSTEM_PROMPT,
    PROMPT_RENDERER_MISTRAL3_INSTRUCT,
    render_prompt_text,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_mistral_renderer() -> None:
    assert render_prompt_text("task", PROMPT_RENDERER_MISTRAL3_INSTRUCT) == (
        "<s>[SYSTEM_PROMPT]"
        + MISTRAL_SYSTEM_PROMPT
        + "[/SYSTEM_PROMPT][INST]task[/INST]"
    )


def test_mistral_config_is_pinned_and_qwen_is_rejected() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/cafd/manufactoria_mistral_cafd_only_v6.yaml").read_text()
    )
    validate_config(config)
    assert config["experiment"]["run_id"] == RUN_ID
    bad = deepcopy(config)
    bad["models"]["student"]["id"] = "Qwen/forbidden"
    try:
        validate_config(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Qwen model was not rejected")
