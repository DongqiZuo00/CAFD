"""Mistral-only entrypoint for the shared CAFD Student trainer."""

from . import train_student
from .mistral_runtime import install_into


if __name__ == "__main__":
    install_into(train_student)
    train_student.main()
