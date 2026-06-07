#!/usr/bin/env python3
"""Single entrypoint for Easy Teacher LoRA training.

Examples:
    python train_easy_teacher_lora.py --teacher numeral
    python train_easy_teacher_lora.py --teacher gravity
    python train_easy_teacher_lora.py --teacher unit
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TEACHER_TYPE = "numeral"  # allowed: numeral, gravity, unit


def load_engine():
    engine_path = Path(__file__).with_name("Teacher-Numeral.py")
    spec = importlib.util.spec_from_file_location("easy_teacher_training_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training engine from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    engine = load_engine()
    if not any(arg == "--teacher" or arg.startswith("--teacher=") for arg in sys.argv[1:]):
        sys.argv.extend(["--teacher", TEACHER_TYPE])
    engine.main()


if __name__ == "__main__":
    main()
