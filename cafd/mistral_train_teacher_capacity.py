"""Mistral-only entrypoint for the shared Teacher capacity trainer."""

from . import train_teacher_capacity
from .mistral_runtime import install_into


if __name__ == "__main__":
    install_into(train_teacher_capacity)
    train_teacher_capacity.main()
