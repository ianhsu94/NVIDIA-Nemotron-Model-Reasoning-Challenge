#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teacher_training.teacher_base import BaseTeacher, TeacherResult
from teacher_training.teachers.bitwise_teacher import BitwiseTeacher
from teacher_training.teachers.cipher_teacher import CipherTeacher
from teacher_training.teachers.gravity_teacher import GravityTeacher
from teacher_training.teachers.numeral_teacher import NumeralTeacher
from teacher_training.teachers.symbol_teacher import SymbolTeacher
from teacher_training.teachers.unit_teacher import UnitTeacher
from teacher_training.utils import CATEGORIES, equivalent_answer, load_all_data, output_root, split_train_validation


def make_teacher(category: str, solver_prior: dict[str, float] | None = None) -> BaseTeacher:
    if category == "numeral":
        return NumeralTeacher()
    if category == "gravity":
        return GravityTeacher()
    if category == "unit":
        return UnitTeacher()
    if category == "bitwise":
        return BitwiseTeacher()
    if category == "cipher":
        return CipherTeacher(solver_prior)
    if category == "symbol":
        return SymbolTeacher(solver_prior)
    raise ValueError(category)


def predict_df(teacher: BaseTeacher, df: pd.DataFrame, category: str) -> list[dict]:
    rows = []
    for row in df.itertuples(index=False):
        result: TeacherResult = teacher.predict_row(row)
        correct = equivalent_answer(result.answer, row.answer, category)
        rows.append(
            {
                "id": str(row.id),
                "category": category,
                "prompt": str(row.prompt),
                "answer": str(row.answer),
                "prediction": result.answer,
                "is_correct": bool(correct),
                "teacher": teacher.name,
                "selected_solver": result.selected_solver,
                "confidence": result.confidence,
                "matched_examples": result.matched_examples,
                "total_examples": result.total_examples,
                "reason": result.reason,
            }
        )
    return rows


def solver_prior_from_rows(rows: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        solver = row.get("selected_solver", "")
        if solver:
            buckets[solver].append(bool(row.get("is_correct", False)))
    return {solver: sum(values) / len(values) for solver, values in buckets.items() if values}


def metrics_from_predictions(predictions: list[dict]) -> dict:
    metrics = {}
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_category[row["category"]].append(row)
    for category in CATEGORIES:
        rows = by_category.get(category, [])
        acc = sum(bool(row["is_correct"]) for row in rows) / len(rows) if rows else 0.0
        entry = {"accuracy": acc, "count": len(rows)}
        if category in {"cipher", "symbol"}:
            solver_buckets: dict[str, list[bool]] = defaultdict(list)
            for row in rows:
                solver_buckets[row["selected_solver"]].append(bool(row["is_correct"]))
            entry["solver_accuracy"] = {
                solver: sum(values) / len(values) for solver, values in sorted(solver_buckets.items()) if values
            }
        metrics[category] = entry
    return metrics


def error_type(row: dict) -> str:
    if row["is_correct"]:
        return ""
    if not str(row["prediction"]).strip():
        return "empty_prediction"
    if row["category"] in {"gravity", "unit"}:
        return "numeric_mismatch"
    if row["matched_examples"] == 0 and row["category"] in {"cipher", "symbol", "bitwise"}:
        return "no_solver_match"
    return "exact_mismatch"


def main() -> None:
    out_root = output_root()
    outputs_dir = out_root / "outputs"
    saved_dir = out_root / "saved_teachers"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data()
    splits = {category: split_train_validation(df) for category, df in data.items()}

    solver_prior: dict[str, dict[str, float]] = {}
    for category in ["cipher", "symbol"]:
        train_df, _ = splits[category]
        teacher = make_teacher(category)
        train_rows = predict_df(teacher, train_df, category)
        solver_prior[category] = solver_prior_from_rows(train_rows)

    all_predictions: list[dict] = []
    for category in CATEGORIES:
        train_df, val_df = splits[category]
        prior = solver_prior.get(category, {})
        teacher = make_teacher(category, prior)
        teacher.fit(train_df)
        all_predictions.extend(predict_df(teacher, val_df, category))

    metrics = metrics_from_predictions(all_predictions)
    errors = [dict(row, error_type=error_type(row)) for row in all_predictions if not row["is_correct"]]

    pd.DataFrame(all_predictions).to_csv(outputs_dir / "teacher_predictions.csv", index=False)
    pd.DataFrame(errors).to_csv(outputs_dir / "error_analysis.csv", index=False)
    with (outputs_dir / "teacher_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (saved_dir / "solver_prior.json").open("w", encoding="utf-8") as f:
        json.dump(solver_prior, f, ensure_ascii=False, indent=2)
    with (saved_dir / "teacher_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_validation_split": "80/20",
                "random_seed": 42,
                "teachers": {
                    "numeral": "Teacher-Numeral",
                    "gravity": "Teacher-Gravity",
                    "unit": "Teacher-Unit",
                    "bitwise": "Teacher-Bitwise",
                    "cipher": "Teacher-Cipher",
                    "symbol": "Teacher-Symbol",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=== Teacher Training Program Complete ===")
    print(f"Output dir: {outputs_dir}")
    for category, entry in metrics.items():
        print(f"{category}: accuracy={entry['accuracy']:.4f}, count={entry['count']}")
    print(f"errors: {len(errors)}")


if __name__ == "__main__":
    main()
