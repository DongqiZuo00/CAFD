"""Mistral-only entrypoint for the shared Student capacity trainer."""

from . import train_capacity
from .mistral_runtime import install_into


if __name__ == "__main__":
    install_into(train_capacity)
    train_capacity.main()
