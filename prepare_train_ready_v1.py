#!/usr/bin/env python3
"""Prepare train_ready_v1 data for the NVIDIA Nemotron reasoning challenge."""

from __future__ import annotations

import os
import random
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 42
SYNTHETIC_ROWS = 3000
SYMBOLIC_SOLVER_TARGET_ROWS = 800
MIN_SYMBOLIC_SOLVER_ROWS = 750

REQUIRED_COLUMNS = ["id", "prompt", "answer", "task_type", "source", "is_replay"]
VALID_TASK_TYPES = {
    "bit_manipulation",
    "gravity",
    "unit_conversion",
    "text_decryption",
    "numeral_system",
    "symbolic_cipher",
    "numeric_equation",
}
VALID_SOURCES = {
    "original",
    "original_replay",
    "synthetic_roman",
    "synthetic_gravity",
    "synthetic_unit_conversion",
    "symbolic_solver_correct",
}
FORBIDDEN_ANSWER_PHRASES = [
    "Answer:",
    "The answer is",
    "Let's solve",
    "step by step",
]

ROMAN_MAP = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


class DataGenerationError(RuntimeError):
    pass


REPORT: dict[str, Any] = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def candidate_roots() -> list[Path]:
    roots = [
        Path("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
    ]
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        roots.append(downloads)
    return roots


def find_input_file(filename: str) -> Path:
    direct_candidates = []
    for root in candidate_roots():
        direct_candidates.append(root / filename)
        direct_candidates.append(root / "nvidia-nemotron-model-reasoning-challenge" / filename)

    for path in direct_candidates:
        if path.exists():
            return path

    for root in candidate_roots():
        if root.exists():
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]

    if filename in {"train.csv", "test.csv"}:
        zip_path = find_competition_zip()
        if zip_path is not None:
            return Path(f"{zip_path}!{filename}")

    raise FileNotFoundError(f"Could not find required input file: {filename}")


def find_competition_zip() -> Path | None:
    names = [
        "nvidia-nemotron-model-reasoning-challenge.zip",
        "nvidia-nemotron-model-reasoning-challenge/*.zip",
    ]
    for root in candidate_roots():
        if not root.exists():
            continue
        for pattern in names:
            for path in root.glob(pattern):
                if path.is_file():
                    try:
                        with zipfile.ZipFile(path) as zf:
                            names_in_zip = set(zf.namelist())
                        if {"train.csv", "test.csv"}.issubset(names_in_zip):
                            return path
                    except zipfile.BadZipFile:
                        continue
    return None


def read_csv_maybe_zip(path: Path) -> pd.DataFrame:
    text = str(path)
    if "!" in text:
        zip_name, member = text.split("!", 1)
        with zipfile.ZipFile(zip_name) as zf:
            with zf.open(member) as handle:
                return pd.read_csv(handle, dtype=str)
    return pd.read_csv(path, dtype=str)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = find_input_file("train.csv")
    test_path = find_input_file("test.csv")
    train_df = read_csv_maybe_zip(train_path)
    test_df = read_csv_maybe_zip(test_path)

    require_columns(train_df, {"id", "prompt", "answer"}, "train.csv")
    require_columns(test_df, {"id", "prompt"}, "test.csv")

    REPORT["train_path"] = str(train_path)
    REPORT["test_path"] = str(test_path)
    REPORT["raw_train_rows"] = len(train_df)
    REPORT["raw_test_rows"] = len(test_df)
    return train_df, test_df


def load_solver_results() -> pd.DataFrame:
    path = find_input_file("solver_results.parquet")
    try:
        solver_df = pd.read_parquet(path)
    except ImportError as exc:
        raise DataGenerationError(
            "Reading solver_results.parquet requires pyarrow or fastparquet. "
            "Install one of them in the runtime before running this script."
        ) from exc
    require_columns(solver_df, {"id", "prompt", "answer", "solver_correct"}, str(path))
    REPORT["solver_results_path"] = str(path)
    REPORT["solver_results_rows"] = len(solver_df)
    return solver_df


def require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise DataGenerationError(f"{label} is missing required columns: {missing}")


def classify_prompt(prompt: str) -> str:
    text = str(prompt)
    low = text.lower()

    if "a secret bit manipulation rule transforms 8-bit binary numbers" in low:
        return "bit_manipulation"
    if "the gravitational constant has been secretly changed" in low:
        return "gravity"
    if "a secret unit conversion is applied to measurements" in low:
        return "unit_conversion"
    if "secret encryption rules are used on text" in low:
        return "text_decryption"
    if "numbers are secretly converted into a different numeral system" in low:
        return "numeral_system"
    if "a secret set of transformation rules is applied to equations" in low:
        return classify_equation_subtype(prompt)
    return "unknown"


def classify_equation_subtype(prompt: str) -> str:
    text = str(prompt)
    examples_block = extract_examples_block(text)
    for expr in re.findall(r"^\s*([^=\n]+?)\s*=", examples_block, flags=re.MULTILINE):
        if re.match(r"^\s*\d+\s*[^0-9\s]\s*\d+\s*$", expr):
            return "numeric_equation"
    return "symbolic_cipher"


def extract_examples_block(text: str) -> str:
    lower = text.lower()
    start = lower.find("examples")
    if start < 0:
        start = lower.find("example")
    if start < 0:
        start = 0
    end = lower.find("now", start)
    if end < 0:
        end = len(text)
    return text[start:end]


def int_to_roman(num: int) -> str:
    if not 1 <= num <= 3999:
        raise ValueError(f"Roman numeral input must be between 1 and 3999: {num}")
    remaining = num
    output: list[str] = []
    for value, symbol in ROMAN_MAP:
        while remaining >= value:
            output.append(symbol)
            remaining -= value
    return "".join(output)


def sample_unique_float(existing: set[str], low: float, high: float, decimals: int) -> str:
    for _ in range(10000):
        value = round(random.uniform(low, high), decimals)
        text = f"{value:.{decimals}f}"
        if text not in existing:
            existing.add(text)
            return text
    raise DataGenerationError("Could not sample a unique value")


def generate_roman_data(n_rows: int) -> pd.DataFrame:
    special_answers = ["IV", "IX", "XL", "XC", "CD", "CM"]
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        if i < 600:
            query_n = roman_to_int_seed_value(special_answers[i % len(special_answers)])
        else:
            query_n = random.randint(1, 3999)

        used = {query_n}
        examples = []
        while len(examples) < 4:
            n = random.randint(1, 3999)
            if n not in used:
                used.add(n)
                examples.append(n)

        lines = [
            "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:",
            *[f"{n} -> {int_to_roman(n)}" for n in examples],
            f"Now, write the number {query_n} in the Wonderland numeral system.",
        ]
        rows.append(
            {
                "id": f"syn_roman_{i:05d}",
                "prompt": "\n".join(lines),
                "answer": int_to_roman(query_n),
                "task_type": "numeral_system",
                "source": "synthetic_roman",
                "is_replay": False,
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def roman_to_int_seed_value(roman: str) -> int:
    return {"IV": 4, "IX": 9, "XL": 40, "XC": 90, "CD": 400, "CM": 900}[roman]


def generate_gravity_data(n_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        g = round(random.uniform(5.0, 15.0), 2)
        used_t: set[str] = set()
        example_ts = [sample_unique_float(used_t, 0.5, 5.0, 2) for _ in range(5)]
        query_t = sample_unique_float(used_t, 0.5, 5.0, 2)

        lines = [
            "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:"
        ]
        for t_text in example_ts:
            t = float(t_text)
            lines.append(f"For t = {t_text}s, distance = {0.5 * g * (t ** 2):.2f} m")
        lines.append(f"Now, determine the falling distance for t = {query_t}s given d = 0.5*g*t^2.")

        qt = float(query_t)
        rows.append(
            {
                "id": f"syn_gravity_{i:05d}",
                "prompt": "\n".join(lines),
                "answer": f"{0.5 * g * (qt ** 2):.2f}",
                "task_type": "gravity",
                "source": "synthetic_gravity",
                "is_replay": False,
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def generate_unit_data(n_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        ratio = round(random.uniform(0.20, 3.00), 4)
        used_x: set[str] = set()
        example_xs = [sample_unique_float(used_x, 1.0, 100.0, 2) for _ in range(5)]
        query_x = sample_unique_float(used_x, 1.0, 100.0, 2)

        lines = ["In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:"]
        for x_text in example_xs:
            x = float(x_text)
            lines.append(f"{x_text} m becomes {x * ratio:.2f}")
        lines.append(f"Now, convert the following measurement: {query_x} m")

        qx = float(query_x)
        rows.append(
            {
                "id": f"syn_unit_{i:05d}",
                "prompt": "\n".join(lines),
                "answer": f"{qx * ratio:.2f}",
                "task_type": "unit_conversion",
                "source": "synthetic_unit_conversion",
                "is_replay": False,
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def prepare_original_data(train_df: pd.DataFrame) -> pd.DataFrame:
    original = train_df[["id", "prompt", "answer"]].copy()
    original["id"] = original["id"].astype(str)
    original["prompt"] = original["prompt"].astype(str)
    original["answer"] = original["answer"].astype(str).str.strip()
    original["task_type"] = original["prompt"].map(classify_prompt)
    original["source"] = "original"
    original["is_replay"] = False

    counts = original["task_type"].value_counts(dropna=False).to_dict()
    REPORT["original_task_type_counts"] = counts
    unknown = original[original["task_type"] == "unknown"]
    if len(unknown) > 0:
        sample = unknown[["id", "prompt"]].head(10).to_dict("records")
        raise DataGenerationError(f"Unknown task_type rows found: count={len(unknown)} sample={sample}")
    return original[REQUIRED_COLUMNS]


def prepare_replay_data(original_df: pd.DataFrame) -> pd.DataFrame:
    replay = original_df.copy()
    replay["id"] = replay["id"].map(lambda value: f"replay_{value}")
    replay["source"] = "original_replay"
    replay["is_replay"] = True
    return replay[REQUIRED_COLUMNS]


def coerce_solver_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value) == 1
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def prepare_symbolic_solver_data(solver_df: pd.DataFrame) -> pd.DataFrame:
    solver = solver_df.copy()
    solver["solver_correct_bool"] = solver["solver_correct"].map(coerce_solver_correct)
    solver = solver[solver["solver_correct_bool"]].copy()
    REPORT["solver_correct_rows_before_answer_check"] = len(solver)

    mismatch_rows = 0
    if "solver_answer" in solver.columns:
        solver["solver_answer"] = solver["solver_answer"].astype(str).str.strip()
        solver["answer_clean"] = solver["answer"].astype(str).str.strip()
        match_mask = solver["solver_answer"] == solver["answer_clean"]
        mismatch_rows = int((~match_mask).sum())
        solver = solver[match_mask].copy()
    REPORT["solver_answer_mismatch_excluded_rows"] = mismatch_rows
    REPORT["solver_correct_rows_after_answer_check"] = len(solver)

    if len(solver) < MIN_SYMBOLIC_SOLVER_ROWS:
        raise DataGenerationError(
            f"Need at least {MIN_SYMBOLIC_SOLVER_ROWS} solver-correct symbolic rows; found {len(solver)}"
        )

    solver = solver.head(SYMBOLIC_SOLVER_TARGET_ROWS).copy()
    result = pd.DataFrame(
        {
            "id": solver["id"].astype(str).map(lambda value: f"symbolic_solver_{value}"),
            "prompt": solver["prompt"].astype(str),
            "answer": solver["answer"].astype(str).str.strip(),
            "task_type": "symbolic_cipher",
            "source": "symbolic_solver_correct",
            "is_replay": False,
        }
    )
    return result[REQUIRED_COLUMNS]


def validate_final_dataset(df: pd.DataFrame) -> None:
    if list(df.columns) != REQUIRED_COLUMNS:
        raise DataGenerationError(f"Final columns must be {REQUIRED_COLUMNS}; got {list(df.columns)}")

    for col in ["id", "prompt", "answer", "task_type", "source"]:
        bad = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if len(bad):
            raise DataGenerationError(f"Column {col} contains blank/null values; first bad ids: {bad['id'].head(20).tolist()}")

    duplicate_ids = df[df["id"].duplicated()]["id"].head(20).tolist()
    if duplicate_ids:
        raise DataGenerationError(f"Duplicate ids found: {duplicate_ids}")

    invalid_tasks = sorted(set(df["task_type"]) - VALID_TASK_TYPES)
    if invalid_tasks:
        raise DataGenerationError(f"Invalid task_type values found: {invalid_tasks}")

    invalid_sources = sorted(set(df["source"]) - VALID_SOURCES)
    if invalid_sources:
        raise DataGenerationError(f"Invalid source values found: {invalid_sources}")

    for phrase in FORBIDDEN_ANSWER_PHRASES:
        mask = df["answer"].astype(str).str.contains(re.escape(phrase), case=False, na=False)
        if mask.any():
            sample = df.loc[mask, ["id", "source", "answer"]].head(20).to_dict("records")
            raise DataGenerationError(f"Forbidden answer phrase {phrase!r} found: {sample}")

    source_counts = df["source"].value_counts()
    max_source_ratio = float(source_counts.max() / len(df))
    if max_source_ratio > 0.40:
        raise DataGenerationError(f"Largest source ratio exceeds 40%: {max_source_ratio:.4f}")

    REPORT["quality_check_status"] = "passed"


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "train_ready_v1.parquet"
    csv_path = output_dir / "train_ready_v1.csv"
    source_counts_path = output_dir / "source_counts.csv"
    task_counts_path = output_dir / "task_type_counts.csv"
    report_path = output_dir / "data_generation_report.txt"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    source_counts = df["source"].value_counts().rename_axis("source").reset_index(name="count")
    task_counts = df["task_type"].value_counts().rename_axis("task_type").reset_index(name="count")
    source_counts.to_csv(source_counts_path, index=False, encoding="utf-8")
    task_counts.to_csv(task_counts_path, index=False, encoding="utf-8")

    REPORT["final_source_counts"] = source_counts.set_index("source")["count"].to_dict()
    REPORT["final_task_type_counts"] = task_counts.set_index("task_type")["count"].to_dict()
    REPORT["final_total_rows"] = len(df)
    REPORT["output_parquet_path"] = str(parquet_path)
    REPORT["output_csv_path"] = str(csv_path)

    lines = ["NVIDIA Nemotron data generation report"]
    for key in [
        "seed",
        "train_path",
        "test_path",
        "solver_results_path",
        "raw_train_rows",
        "raw_test_rows",
        "original_task_type_counts",
        "solver_results_rows",
        "solver_correct_rows_before_answer_check",
        "solver_correct_rows_after_answer_check",
        "solver_answer_mismatch_excluded_rows",
        "synthetic_roman_rows",
        "synthetic_gravity_rows",
        "synthetic_unit_conversion_rows",
        "final_source_counts",
        "final_task_type_counts",
        "final_total_rows",
        "output_parquet_path",
        "output_csv_path",
        "quality_check_status",
    ]:
        lines.append(f"{key}: {REPORT.get(key)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_output_dir() -> Path:
    env_dir = os.getenv("NEMOTRON_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "nemotron_data_ready"
    return Path.cwd() / "nemotron_data_ready"


def main() -> None:
    set_seed(SEED)
    REPORT.clear()
    REPORT["seed"] = SEED

    train_df, _test_df = load_raw_data()
    solver_df = load_solver_results()

    original = prepare_original_data(train_df)
    replay = prepare_replay_data(original)
    roman_df = generate_roman_data(SYNTHETIC_ROWS)
    gravity_df = generate_gravity_data(SYNTHETIC_ROWS)
    unit_df = generate_unit_data(SYNTHETIC_ROWS)
    symbolic_df = prepare_symbolic_solver_data(solver_df)

    REPORT["synthetic_roman_rows"] = len(roman_df)
    REPORT["synthetic_gravity_rows"] = len(gravity_df)
    REPORT["synthetic_unit_conversion_rows"] = len(unit_df)

    final_df = pd.concat([original, replay, roman_df, gravity_df, unit_df, symbolic_df], ignore_index=True)
    final_df = final_df[REQUIRED_COLUMNS]
    validate_final_dataset(final_df)

    output_dir = resolve_output_dir()
    write_outputs(final_df, output_dir)

    print("Data generation completed successfully.")
    print(f"Output directory: {output_dir}")
    print(f"train_ready_v1.parquet: {output_dir / 'train_ready_v1.parquet'}")
    print(f"train_ready_v1.csv: {output_dir / 'train_ready_v1.csv'}")
    print(f"Final rows: {len(final_df)}")
    print("Source counts:")
    print(final_df["source"].value_counts().to_string())
    print("Task type counts:")
    print(final_df["task_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
