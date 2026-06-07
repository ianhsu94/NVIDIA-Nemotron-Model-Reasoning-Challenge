#!/usr/bin/env python3
"""Utilities for splitting easy teacher datasets from train_ready_v1."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 42
FORMAT_REPLAY_ROWS = 1000

INPUT_COLUMNS = ["id", "prompt", "answer", "task_type", "source", "is_replay"]
OUTPUT_COLUMNS = [
    "id",
    "prompt",
    "answer",
    "task_type",
    "source",
    "is_replay",
    "teacher_target",
    "teacher_data_role",
]
REQUIRED_NONEMPTY_COLUMNS = ["id", "prompt", "answer", "task_type", "source", "teacher_target", "teacher_data_role"]
VALID_TEACHER_TARGETS = {"teacher_numeral", "teacher_gravity", "teacher_unit"}
VALID_TEACHER_ROLES = {"main", "format_replay"}
FORBIDDEN_ANSWER_PHRASES = ["Answer:", "The answer is", "Let's solve", "step by step", "```"]


class TeacherDataSplitError(RuntimeError):
    pass


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_train_ready_file() -> Path:
    env_path = os.getenv("TRAIN_READY_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"TRAIN_READY_PATH does not exist: {path}")

    candidates = [
        Path("/kaggle/working/nemotron_data_ready/train_ready_v1.parquet"),
        Path.cwd() / "nemotron_data_ready" / "train_ready_v1.parquet",
        Path.cwd() / "train_ready_v1.parquet",
    ]
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").rglob("train_ready_v1.parquet"))

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find train_ready_v1.parquet")


def load_train_ready(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise TeacherDataSplitError(
            "Reading train_ready_v1.parquet requires pyarrow or fastparquet."
        ) from exc

    validate_input_df(df)
    for col in ["id", "prompt", "answer", "task_type", "source"]:
        df[col] = df[col].astype(str)
    if "is_replay" in df.columns:
        df["is_replay"] = df["is_replay"].astype(bool)
    return df


def validate_input_df(df: pd.DataFrame) -> None:
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise TeacherDataSplitError(f"train_ready_v1.parquet missing columns: {missing}")

    for col in ["id", "prompt", "answer", "task_type", "source"]:
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if bad.any():
            sample = df.loc[bad, "id"].head(20).astype(str).tolist()
            raise TeacherDataSplitError(f"Input column {col} has blank/null values. Sample ids: {sample}")

    for phrase in FORBIDDEN_ANSWER_PHRASES:
        mask = df["answer"].astype(str).str.contains(re.escape(phrase), case=False, na=False)
        if mask.any():
            sample = df.loc[mask, ["id", "task_type", "source", "answer"]].head(20).to_dict("records")
            raise TeacherDataSplitError(f"Input answer contains forbidden phrase {phrase!r}: {sample}")


def build_teacher_dataset(
    df: pd.DataFrame,
    teacher_name: str,
    main_task_type: str,
    allowed_main_sources: list[str],
    replay_exclude_task_type: str,
) -> pd.DataFrame:
    if teacher_name not in VALID_TEACHER_TARGETS:
        raise TeacherDataSplitError(f"Invalid teacher_name: {teacher_name}")

    main = df[(df["task_type"] == main_task_type) & (df["source"].isin(allowed_main_sources))].copy()
    if len(main) == 0:
        raise TeacherDataSplitError(f"No main rows found for {teacher_name}")

    replay_pool = df[(df["source"] == "original") & (df["task_type"] != replay_exclude_task_type)].copy()
    if len(replay_pool) == 0:
        raise TeacherDataSplitError(f"No format replay rows available for {teacher_name}")
    replay_n = min(FORMAT_REPLAY_ROWS, len(replay_pool))
    replay = replay_pool.sample(n=replay_n, random_state=SEED).copy()

    main["teacher_target"] = teacher_name
    main["teacher_data_role"] = "main"
    main["id"] = main["id"].astype(str).map(lambda value: f"{teacher_name}_main_{value}")

    replay["teacher_target"] = teacher_name
    replay["teacher_data_role"] = "format_replay"
    replay["id"] = replay["id"].astype(str).map(lambda value: f"{teacher_name}_replay_{value}")

    teacher_df = pd.concat([main, replay], ignore_index=True)
    teacher_df = teacher_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    teacher_df = teacher_df[OUTPUT_COLUMNS]

    if len(teacher_df) < 2000:
        raise TeacherDataSplitError(f"{teacher_name} final rows must be at least 2000; got {len(teacher_df)}")
    return teacher_df


def validate_teacher_dataset(df: pd.DataFrame, teacher_name: str) -> None:
    if list(df.columns) != OUTPUT_COLUMNS:
        raise TeacherDataSplitError(f"Output columns mismatch. Got {list(df.columns)}")

    for col in REQUIRED_NONEMPTY_COLUMNS:
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if bad.any():
            sample = df.loc[bad, "id"].head(20).astype(str).tolist()
            raise TeacherDataSplitError(f"Output column {col} has blank/null values. Sample ids: {sample}")

    duplicate_ids = df.loc[df["id"].duplicated(), "id"].head(20).tolist()
    if duplicate_ids:
        raise TeacherDataSplitError(f"Duplicate ids found: {duplicate_ids}")

    if set(df["teacher_target"]) != {teacher_name}:
        raise TeacherDataSplitError(f"teacher_target must be only {teacher_name}")

    invalid_roles = sorted(set(df["teacher_data_role"]) - VALID_TEACHER_ROLES)
    if invalid_roles:
        raise TeacherDataSplitError(f"Invalid teacher_data_role values: {invalid_roles}")

    for phrase in FORBIDDEN_ANSWER_PHRASES:
        mask = df["answer"].astype(str).str.contains(re.escape(phrase), case=False, na=False)
        if mask.any():
            sample = df.loc[mask, ["id", "source", "answer"]].head(20).to_dict("records")
            raise TeacherDataSplitError(f"Output answer contains forbidden phrase {phrase!r}: {sample}")


def default_output_dir(teacher_name: str) -> Path:
    base = Path("/kaggle/working/easy_teacher_data") if Path("/kaggle/working").exists() else Path.cwd() / "easy_teacher_data"
    return base / teacher_name


def write_outputs(df: pd.DataFrame, output_dir: Path, teacher_name: str, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{teacher_name}_train.parquet"
    csv_path = output_dir / f"{teacher_name}_train.csv"
    source_counts_path = output_dir / f"{teacher_name}_source_counts.csv"
    task_counts_path = output_dir / f"{teacher_name}_task_type_counts.csv"
    report_path = output_dir / f"{teacher_name}_data_report.txt"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    source_counts = df["source"].value_counts().rename_axis("source").reset_index(name="count")
    task_counts = df["task_type"].value_counts().rename_axis("task_type").reset_index(name="count")
    source_counts.to_csv(source_counts_path, index=False, encoding="utf-8")
    task_counts.to_csv(task_counts_path, index=False, encoding="utf-8")

    report = dict(report)
    report.update(
        {
            "source_counts": source_counts.set_index("source")["count"].to_dict(),
            "task_type_counts": task_counts.set_index("task_type")["count"].to_dict(),
            "output_parquet_path": str(parquet_path),
            "output_csv_path": str(csv_path),
            "quality_check_status": "passed",
        }
    )
    write_report(report_path, report)


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    keys = [
        "program_name",
        "seed",
        "train_ready_path",
        "raw_train_ready_rows",
        "teacher_name",
        "main_filter_rule",
        "format_replay_filter_rule",
        "main_rows",
        "format_replay_rows",
        "final_rows",
        "source_counts",
        "task_type_counts",
        "output_parquet_path",
        "output_csv_path",
        "quality_check_status",
    ]
    lines = ["Data split completed successfully."]
    lines.extend(f"{key}: {report.get(key)}" for key in keys)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_teacher_split(
    *,
    program_name: str,
    teacher_name: str,
    main_task_type: str,
    allowed_main_sources: list[str],
) -> None:
    set_seed(SEED)
    train_ready_path = find_train_ready_file()
    raw_df = load_train_ready(train_ready_path)
    teacher_df = build_teacher_dataset(
        raw_df,
        teacher_name=teacher_name,
        main_task_type=main_task_type,
        allowed_main_sources=allowed_main_sources,
        replay_exclude_task_type=main_task_type,
    )
    validate_teacher_dataset(teacher_df, teacher_name)

    main_rows = int((teacher_df["teacher_data_role"] == "main").sum())
    replay_rows = int((teacher_df["teacher_data_role"] == "format_replay").sum())
    report = {
        "program_name": program_name,
        "seed": SEED,
        "train_ready_path": str(train_ready_path),
        "raw_train_ready_rows": len(raw_df),
        "teacher_name": teacher_name,
        "main_filter_rule": f'task_type == "{main_task_type}" and source in {allowed_main_sources}',
        "format_replay_filter_rule": f'source == "original" and task_type != "{main_task_type}"',
        "main_rows": main_rows,
        "format_replay_rows": replay_rows,
        "final_rows": len(teacher_df),
    }

    output_dir = default_output_dir(teacher_name)
    write_outputs(teacher_df, output_dir, teacher_name, report)

    print("Data split completed successfully.")
    print(f"Teacher: {teacher_name}")
    print(f"Output directory: {output_dir}")
    print(f"Output parquet: {output_dir / f'{teacher_name}_train.parquet'}")
    print(f"Output csv: {output_dir / f'{teacher_name}_train.csv'}")
    print(f"Final rows: {len(teacher_df)}")
    print("Source counts:")
    print(teacher_df["source"].value_counts().to_string())
    print("Task type counts:")
    print(teacher_df["task_type"].value_counts().to_string())
