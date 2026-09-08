"""Mistral-only entrypoint for the shared Teacher RLVR trainer."""

from . import train_teacher
from .configuration import load_experiment_config
from .mistral_disjoint import teacher_rlvr_rows
from .mistral_runtime import V7_PROTOCOL, install_into


_ORIGINAL_LOAD_ROWS = train_teacher.load_rows


def _load_rows(root, split):
    config = load_experiment_config(root)
    if str(config["protocol"]) == V7_PROTOCOL and split == "train":
        return teacher_rlvr_rows(root, config)
    return _ORIGINAL_LOAD_ROWS(root, split)


if __name__ == "__main__":
    install_into(train_teacher)
    train_teacher.load_rows = _load_rows
    train_teacher.main()
