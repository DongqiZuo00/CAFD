"""Mistral-only entrypoint for one-time frozen CAFD evaluation."""

from . import evaluate_condition
from .mistral_runtime import install_into


if __name__ == "__main__":
    install_into(evaluate_condition)
    evaluate_condition.main()
