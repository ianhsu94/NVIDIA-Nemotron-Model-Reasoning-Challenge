#!/usr/bin/env python3
"""Train Teacher-Gravity LoRA adapter for the Nemotron Reasoning Challenge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_engine():
    if "__file__" in globals():
        engine_path = Path(__file__).with_name("Teacher-Numeral.py")
    else:
        engine_path = Path("/kaggle/working/Teacher-Numeral.py")
        if not engine_path.exists():
            engine_path = Path.cwd() / "Teacher-Numeral.py"
    spec = importlib.util.spec_from_file_location("teacher_training_engine", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training engine from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    engine = load_engine()
    if "--teacher" not in sys.argv:
        sys.argv.extend(["--teacher", "gravity"])
    engine.main()


if __name__ == "__main__":
    main()
