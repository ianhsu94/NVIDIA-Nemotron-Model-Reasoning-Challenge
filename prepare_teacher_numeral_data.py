#!/usr/bin/env python3
"""Prepare Teacher-Numeral easy training data."""

from easy_teacher_data_utils import run_teacher_split


def main() -> None:
    run_teacher_split(
        program_name="prepare_teacher_numeral_data.py",
        teacher_name="teacher_numeral",
        main_task_type="numeral_system",
        allowed_main_sources=["original", "original_replay", "synthetic_roman"],
    )


if __name__ == "__main__":
    main()
