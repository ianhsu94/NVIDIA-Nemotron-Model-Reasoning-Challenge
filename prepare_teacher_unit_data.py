#!/usr/bin/env python3
"""Prepare Teacher-Unit easy training data."""

from easy_teacher_data_utils import run_teacher_split


def main() -> None:
    run_teacher_split(
        program_name="prepare_teacher_unit_data.py",
        teacher_name="teacher_unit",
        main_task_type="unit_conversion",
        allowed_main_sources=["original", "original_replay", "synthetic_unit_conversion"],
    )


if __name__ == "__main__":
    main()
