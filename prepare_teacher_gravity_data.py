#!/usr/bin/env python3
"""Prepare Teacher-Gravity easy training data."""

from easy_teacher_data_utils import run_teacher_split


def main() -> None:
    run_teacher_split(
        program_name="prepare_teacher_gravity_data.py",
        teacher_name="teacher_gravity",
        main_task_type="gravity",
        allowed_main_sources=["original", "original_replay", "synthetic_gravity"],
    )


if __name__ == "__main__":
    main()
