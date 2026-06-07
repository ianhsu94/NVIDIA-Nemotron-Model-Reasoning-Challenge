#!/usr/bin/env python3
"""Split train_ready_v1.parquet into 14 teacher training datasets."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SEED = 42
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
    "teacher_strategy",
]
VALID_ROLES = {"main", "format_replay", "strategy_replay"}
FORBIDDEN_ANSWER_PHRASES = ["Answer:", "The answer is", "Let's solve", "step by step", "```"]


class TeacherSplitError(RuntimeError):
    pass


def set_seed(seed: int) -> None:
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
    if Path("/kaggle/working").exists():
        candidates.extend(Path("/kaggle/working").rglob("train_ready_v1.parquet"))

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find train_ready_v1.parquet")


def load_train_ready(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise TeacherSplitError("Reading parquet requires pyarrow or fastparquet.") from exc
    validate_input_df(df)
    for col in ["id", "prompt", "answer", "task_type", "source"]:
        df[col] = df[col].astype(str)
    df["is_replay"] = df["is_replay"].astype(bool)
    return df


def validate_input_df(df: pd.DataFrame) -> None:
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise TeacherSplitError(f"train_ready_v1.parquet missing columns: {missing}")
    for col in ["id", "prompt", "answer", "task_type", "source"]:
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if bad.any():
            sample = df.loc[bad, "id"].head(20).astype(str).tolist()
            raise TeacherSplitError(f"Input column {col} has blank/null values. Sample ids: {sample}")
    for phrase in FORBIDDEN_ANSWER_PHRASES:
        mask = df["answer"].astype(str).str.contains(re.escape(phrase), case=False, na=False)
        if mask.any():
            sample = df.loc[mask, ["id", "source", "answer"]].head(20).to_dict("records")
            raise TeacherSplitError(f"Input answer contains forbidden phrase {phrase!r}: {sample}")


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    used = min(n, len(df))
    return df.sample(n=used, random_state=seed).copy()


def add_teacher_columns(
    df: pd.DataFrame,
    teacher_target: str,
    role: str,
    strategy: str,
    id_prefix: str,
) -> pd.DataFrame:
    result = df.copy()
    result["id"] = result["id"].astype(str).map(lambda value: f"{id_prefix}_{value}")
    result["teacher_target"] = teacher_target
    result["teacher_data_role"] = role
    result["teacher_strategy"] = strategy
    return result[OUTPUT_COLUMNS]


def combine_parts(*parts: pd.DataFrame) -> pd.DataFrame:
    nonempty = [part for part in parts if len(part) > 0]
    if not nonempty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(nonempty, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def original_except(df: pd.DataFrame, excluded_tasks: set[str]) -> pd.DataFrame:
    return df[(df["source"] == "original") & (~df["task_type"].isin(excluded_tasks))].copy()


def build_teacher_general(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["source"].isin(["original", "original_replay"])].copy()
    return add_teacher_columns(
        main,
        "teacher_general",
        "main",
        "general_original_distribution",
        "teacher_general_main",
    )


def build_teacher_numeral(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "numeral_system") & (df["source"].isin(["original", "original_replay", "synthetic_roman"]))].copy()
    replay = sample_df(original_except(df, {"numeral_system"}), 1000, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_numeral", "main", "numeral_main_roman_rule", "teacher_numeral_main"),
        add_teacher_columns(replay, "teacher_numeral", "format_replay", "numeral_format_replay_other_original", "teacher_numeral_replay"),
    )


def build_teacher_gravity(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "gravity") & (df["source"].isin(["original", "original_replay", "synthetic_gravity"]))].copy()
    replay = sample_df(original_except(df, {"gravity"}), 1000, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_gravity", "main", "gravity_main_formula", "teacher_gravity_main"),
        add_teacher_columns(replay, "teacher_gravity", "format_replay", "gravity_format_replay_other_original", "teacher_gravity_replay"),
    )


def build_teacher_unit(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "unit_conversion") & (df["source"].isin(["original", "original_replay", "synthetic_unit_conversion"]))].copy()
    replay = sample_df(original_except(df, {"unit_conversion"}), 1000, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_unit", "main", "unit_main_ratio_conversion", "teacher_unit_main"),
        add_teacher_columns(replay, "teacher_unit", "format_replay", "unit_format_replay_other_original", "teacher_unit_replay"),
    )


def build_teacher_symbolic_solver(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "symbolic_cipher") & (df["source"].isin(["original", "original_replay", "symbolic_solver_correct"]))].copy()
    replay = sample_df(original_except(df, {"symbolic_cipher"}), 1000, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_symbolic_solver", "main", "symbolic_solver_correct_focus", "teacher_symbolic_solver_main"),
        add_teacher_columns(replay, "teacher_symbolic_solver", "format_replay", "symbolic_solver_format_replay_other_original", "teacher_symbolic_solver_replay"),
    )


def build_teacher_symbolic_general(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "symbolic_cipher") & (df["source"].isin(["original", "original_replay"]))].copy()
    numeric = sample_df(df[(df["task_type"] == "numeric_equation") & (df["source"] == "original")], 500, SEED)
    replay = sample_df(original_except(df, {"symbolic_cipher", "numeric_equation"}), 1000, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_symbolic_general", "main", "symbolic_general_original_focus", "teacher_symbolic_general_main"),
        add_teacher_columns(numeric, "teacher_symbolic_general", "strategy_replay", "symbolic_general_numeric_equation_replay", "teacher_symbolic_general_numeric_replay"),
        add_teacher_columns(replay, "teacher_symbolic_general", "format_replay", "symbolic_general_format_replay_other_original", "teacher_symbolic_general_replay"),
    )


def build_teacher_text_mapping(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "text_decryption") & (df["source"].isin(["original", "original_replay"]))].copy()
    replay = sample_df(original_except(df, {"text_decryption"}), 1000, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_text_mapping", "main", "text_mapping_original_focus", "teacher_text_mapping_main"),
        add_teacher_columns(replay, "teacher_text_mapping", "format_replay", "text_mapping_format_replay_other_original", "teacher_text_mapping_replay"),
    )


def build_teacher_text_format(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "text_decryption") & (df["source"] == "original")].copy()
    replay = sample_df(original_except(df, {"text_decryption"}), 1500, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_text_format", "main", "text_format_original_focus", "teacher_text_format_main"),
        add_teacher_columns(replay, "teacher_text_format", "strategy_replay", "text_format_all_task_answer_format_replay", "teacher_text_format_replay"),
    )


def build_teacher_numeric_arithmetic(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "numeric_equation") & (df["source"].isin(["original", "original_replay"]))].copy()
    math_replay = sample_df(df[(df["task_type"].isin(["gravity", "unit_conversion"])) & (df["source"] == "original")], 800, SEED)
    replay = sample_df(original_except(df, {"numeric_equation", "gravity", "unit_conversion"}), 800, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_numeric_arithmetic", "main", "numeric_arithmetic_original_focus", "teacher_numeric_arithmetic_main"),
        add_teacher_columns(math_replay, "teacher_numeric_arithmetic", "strategy_replay", "numeric_arithmetic_gravity_unit_replay", "teacher_numeric_arithmetic_math_replay"),
        add_teacher_columns(replay, "teacher_numeric_arithmetic", "format_replay", "numeric_arithmetic_format_replay_other_original", "teacher_numeric_arithmetic_replay"),
    )


def build_teacher_numeric_digitstring(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "numeric_equation") & (df["source"].isin(["original", "original_replay"]))].copy()
    bit_replay = sample_df(df[(df["task_type"] == "bit_manipulation") & (df["source"] == "original")], 800, SEED)
    replay = sample_df(original_except(df, {"numeric_equation", "bit_manipulation"}), 800, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_numeric_digitstring", "main", "numeric_digitstring_original_focus", "teacher_numeric_digitstring_main"),
        add_teacher_columns(bit_replay, "teacher_numeric_digitstring", "strategy_replay", "numeric_digitstring_bit_pattern_replay", "teacher_numeric_digitstring_bit_replay"),
        add_teacher_columns(replay, "teacher_numeric_digitstring", "format_replay", "numeric_digitstring_format_replay_other_original", "teacher_numeric_digitstring_replay"),
    )


def build_teacher_numeric_mixed(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "numeric_equation") & (df["source"].isin(["original", "original_replay"]))].copy()
    symbolic_replay = sample_df(df[(df["task_type"] == "symbolic_cipher") & (df["source"].isin(["original", "symbolic_solver_correct"]))], 1000, SEED)
    replay = sample_df(original_except(df, {"numeric_equation", "symbolic_cipher"}), 800, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_numeric_mixed", "main", "numeric_mixed_original_focus", "teacher_numeric_mixed_main"),
        add_teacher_columns(symbolic_replay, "teacher_numeric_mixed", "strategy_replay", "numeric_mixed_symbolic_replay", "teacher_numeric_mixed_symbolic_replay"),
        add_teacher_columns(replay, "teacher_numeric_mixed", "format_replay", "numeric_mixed_format_replay_other_original", "teacher_numeric_mixed_replay"),
    )


def build_teacher_bit_original(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "bit_manipulation") & (df["source"].isin(["original", "original_replay"]))].copy()
    replay = sample_df(original_except(df, {"bit_manipulation"}), 1000, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_bit_original", "main", "bit_original_distribution_focus", "teacher_bit_original_main"),
        add_teacher_columns(replay, "teacher_bit_original", "format_replay", "bit_original_format_replay_other_original", "teacher_bit_original_replay"),
    )


def build_teacher_bit_operation(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "bit_manipulation") & (df["source"].isin(["original", "original_replay"]))].copy()
    equation = sample_df(df[(df["task_type"].isin(["numeric_equation", "symbolic_cipher"])) & (df["source"] == "original")], 1000, SEED)
    replay = sample_df(original_except(df, {"bit_manipulation", "numeric_equation", "symbolic_cipher"}), 800, SEED)
    return combine_parts(
        add_teacher_columns(main, "teacher_bit_operation", "main", "bit_operation_original_focus", "teacher_bit_operation_main"),
        add_teacher_columns(equation, "teacher_bit_operation", "strategy_replay", "bit_operation_equation_pattern_replay", "teacher_bit_operation_equation_replay"),
        add_teacher_columns(replay, "teacher_bit_operation", "format_replay", "bit_operation_format_replay_other_original", "teacher_bit_operation_replay"),
    )


def build_teacher_bit_augmented(df: pd.DataFrame) -> pd.DataFrame:
    main = df[(df["task_type"] == "bit_manipulation") & (df["source"].isin(["original", "original_replay"]))].copy()
    replay_parts = []
    for task_type, group in df[(df["source"] == "original") & (df["task_type"] != "bit_manipulation")].groupby("task_type"):
        replay_parts.append(sample_df(group, 300, SEED + len(replay_parts)))
    balanced = pd.concat(replay_parts, ignore_index=True) if replay_parts else df.iloc[0:0].copy()
    return combine_parts(
        add_teacher_columns(main, "teacher_bit_augmented", "main", "bit_augmented_original_focus", "teacher_bit_augmented_main"),
        add_teacher_columns(balanced, "teacher_bit_augmented", "strategy_replay", "bit_augmented_balanced_other_task_replay", "teacher_bit_augmented_balanced_replay"),
    )


TEACHER_BUILDERS: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
    ("teacher_general", build_teacher_general),
    ("teacher_numeral", build_teacher_numeral),
    ("teacher_gravity", build_teacher_gravity),
    ("teacher_unit", build_teacher_unit),
    ("teacher_symbolic_solver", build_teacher_symbolic_solver),
    ("teacher_symbolic_general", build_teacher_symbolic_general),
    ("teacher_text_mapping", build_teacher_text_mapping),
    ("teacher_text_format", build_teacher_text_format),
    ("teacher_numeric_arithmetic", build_teacher_numeric_arithmetic),
    ("teacher_numeric_digitstring", build_teacher_numeric_digitstring),
    ("teacher_numeric_mixed", build_teacher_numeric_mixed),
    ("teacher_bit_original", build_teacher_bit_original),
    ("teacher_bit_operation", build_teacher_bit_operation),
    ("teacher_bit_augmented", build_teacher_bit_augmented),
]


def validate_teacher_df(df: pd.DataFrame, teacher_target: str) -> None:
    if list(df.columns) != OUTPUT_COLUMNS:
        raise TeacherSplitError(f"{teacher_target} columns mismatch: {list(df.columns)}")
    if len(df) < 1000:
        raise TeacherSplitError(f"{teacher_target} final rows must be at least 1000; got {len(df)}")
    main_rows = int((df["teacher_data_role"] == "main").sum())
    if main_rows < 200:
        raise TeacherSplitError(f"{teacher_target} main rows must be at least 200; got {main_rows}")
    for col in ["id", "prompt", "answer", "task_type", "source", "teacher_target", "teacher_data_role", "teacher_strategy"]:
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if bad.any():
            sample = df.loc[bad, "id"].head(20).astype(str).tolist()
            raise TeacherSplitError(f"{teacher_target} column {col} has blanks. Sample ids: {sample}")
    dupes = df.loc[df["id"].duplicated(), "id"].head(20).tolist()
    if dupes:
        raise TeacherSplitError(f"{teacher_target} duplicate ids: {dupes}")
    if set(df["teacher_target"]) != {teacher_target}:
        raise TeacherSplitError(f"{teacher_target} has wrong teacher_target values: {sorted(set(df['teacher_target']))}")
    invalid_roles = sorted(set(df["teacher_data_role"]) - VALID_ROLES)
    if invalid_roles:
        raise TeacherSplitError(f"{teacher_target} invalid roles: {invalid_roles}")
    for phrase in FORBIDDEN_ANSWER_PHRASES:
        mask = df["answer"].astype(str).str.contains(re.escape(phrase), case=False, na=False)
        if mask.any():
            sample = df.loc[mask, ["id", "teacher_target", "source", "answer"]].head(20).to_dict("records")
            raise TeacherSplitError(f"{teacher_target} answer contains forbidden phrase {phrase!r}: {sample}")


def output_root() -> Path:
    env_dir = os.getenv("TEACHER_OUTPUT_ROOT")
    if env_dir:
        return Path(env_dir)
    return Path("/kaggle/working/teacher_train_data_v1") if Path("/kaggle/working").exists() else Path.cwd() / "teacher_train_data_v1"


def write_teacher_outputs(
    df: pd.DataFrame,
    teacher_target: str,
    root: Path,
    train_ready_path: Path,
    raw_train_ready_rows: int,
) -> dict[str, Any]:
    out_dir = root / teacher_target
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{teacher_target}_train.parquet"
    csv_path = out_dir / f"{teacher_target}_train.csv"
    report_path = out_dir / f"{teacher_target}_data_report.txt"
    source_counts_path = out_dir / f"{teacher_target}_source_counts.csv"
    task_counts_path = out_dir / f"{teacher_target}_task_type_counts.csv"
    role_counts_path = out_dir / f"{teacher_target}_role_counts.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    source_counts = df["source"].value_counts().rename_axis("source").reset_index(name="count")
    task_counts = df["task_type"].value_counts().rename_axis("task_type").reset_index(name="count")
    role_counts = df["teacher_data_role"].value_counts().rename_axis("teacher_data_role").reset_index(name="count")
    source_counts.to_csv(source_counts_path, index=False, encoding="utf-8")
    task_counts.to_csv(task_counts_path, index=False, encoding="utf-8")
    role_counts.to_csv(role_counts_path, index=False, encoding="utf-8")

    stats = {
        "teacher_target": teacher_target,
        "train_ready_path": str(train_ready_path),
        "raw_train_ready_rows": raw_train_ready_rows,
        "final_rows": len(df),
        "main_rows": int((df["teacher_data_role"] == "main").sum()),
        "format_replay_rows": int((df["teacher_data_role"] == "format_replay").sum()),
        "strategy_replay_rows": int((df["teacher_data_role"] == "strategy_replay").sum()),
        "source_counts": source_counts.set_index("source")["count"].to_dict(),
        "task_type_counts": task_counts.set_index("task_type")["count"].to_dict(),
        "role_counts": role_counts.set_index("teacher_data_role")["count"].to_dict(),
        "output_parquet_path": str(parquet_path),
        "output_csv_path": str(csv_path),
        "quality_check_status": "passed",
    }
    lines = ["Teacher data split completed successfully."]
    for key in [
        "teacher_target",
        "train_ready_path",
        "raw_train_ready_rows",
        "final_rows",
        "main_rows",
        "format_replay_rows",
        "strategy_replay_rows",
        "source_counts",
        "task_type_counts",
        "role_counts",
        "output_parquet_path",
        "output_csv_path",
        "quality_check_status",
    ]:
        lines.append(f"{key}: {stats.get(key)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def write_summary(all_stats: list[dict[str, Any]], root: Path, train_ready_path: Path, raw_rows: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "teacher_target": stat["teacher_target"],
                "rows": stat["final_rows"],
                "main_rows": stat["main_rows"],
                "format_replay_rows": stat["format_replay_rows"],
                "strategy_replay_rows": stat["strategy_replay_rows"],
            }
            for stat in all_stats
        ]
    )
    summary.to_csv(root / "all_teacher_data_summary.csv", index=False, encoding="utf-8")
    report_lines = [
        "14-teacher data split report",
        f"seed: {SEED}",
        f"train_ready_path: {train_ready_path}",
        f"raw_train_ready_rows: {raw_rows}",
        f"output_root: {root}",
        f"all_teacher_summary: {summary.to_dict('records')}",
        "quality_check_status: passed",
    ]
    (root / "all_teacher_data_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    train_ready_path = find_train_ready_file()
    df = load_train_ready(train_ready_path)
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict[str, Any]] = []
    for teacher_target, builder in TEACHER_BUILDERS:
        teacher_df = builder(df)
        validate_teacher_df(teacher_df, teacher_target)
        stats = write_teacher_outputs(teacher_df, teacher_target, root, train_ready_path, len(df))
        all_stats.append(stats)

    write_summary(all_stats, root, train_ready_path, len(df))

    print("All teacher data splits completed successfully.")
    print(f"Output root: {root}")
    print("Generated teacher datasets:")
    for stat in all_stats:
        print(f"{stat['teacher_target']}: {stat['final_rows']}")


if __name__ == "__main__":
    main()
