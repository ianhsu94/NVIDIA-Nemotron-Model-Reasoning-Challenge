from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 42
CATEGORIES = ["numeral", "gravity", "unit", "bitwise", "cipher", "symbol"]
DATA_FILES = {
    "numeral": "numeral_train.csv",
    "gravity": "gravity_train.csv",
    "unit": "unit_train.csv",
    "bitwise": "bitwise_train.csv",
    "cipher": "cipher_full.csv",
    "symbol": "symbol_full.csv",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/teacher_training")
    return project_root() / "teacher_training"


def training_data_dir() -> Path:
    candidates = [
        Path("/kaggle/working/training_data"),
        project_root() / "training_data",
        Path.cwd() / "training_data",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find training_data directory. Run generate_training_data.py first.")


def load_category_data(category: str) -> pd.DataFrame:
    data_dir = training_data_dir()
    path = data_dir / DATA_FILES[category]
    if not path.exists():
        fallback = data_dir / DATA_FILES[category].replace("_full", "_train")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"id": str, "prompt": str, "answer": str})
    if "category" not in df.columns:
        df["category"] = category
    return df


def load_all_data() -> dict[str, pd.DataFrame]:
    return {category: load_category_data(category) for category in CATEGORIES}


def split_train_validation(df: pd.DataFrame, seed: int = SEED, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_n = max(1, int(round(len(shuffled) * val_frac)))
    val = shuffled.iloc[:val_n].copy()
    train = shuffled.iloc[val_n:].copy()
    if set(train["id"]).intersection(set(val["id"])):
        raise AssertionError("id overlap between train and validation")
    return train, val


def parse_examples(row: Any) -> list[tuple[str, str]]:
    value = getattr(row, "examples_json", None)
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        try:
            parsed = json.loads(value)
            return [(str(a), str(b)) for a, b in parsed]
        except Exception:
            pass

    examples = []
    prompt = str(getattr(row, "prompt", ""))
    category = str(getattr(row, "category", ""))
    for raw in prompt.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("now,"):
            continue
        if line.endswith(":"):
            continue
        m = re.match(r"^(.+?)\s*->\s*(.+?)$", line)
        if not m and category in {"symbol", "gravity"}:
            m = re.match(r"^(.+?)\s*=\s*(.+?)$", line)
        if not m and category == "unit":
            m = re.match(r"^(.+?)\s+becomes\s+(.+?)$", line, re.I)
        if not m and category == "gravity":
            m = re.match(r"^For\s+(.+?),\s*distance\s*=\s*(.+?)$", line, re.I)
        if m:
            examples.append((m.group(1).strip(), m.group(2).strip()))
    return examples


def target_text(row: Any) -> str:
    value = getattr(row, "target_text", "")
    if value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip():
        return str(value).strip()
    prompt = str(getattr(row, "prompt", ""))
    patterns = [
        r"Now,\s*determine the output for:\s*(.+)",
        r"Now,\s*decrypt the following text:\s*(.+)",
        r"Now,\s*determine the result for:\s*(.+)",
        r"Now,\s*convert the following measurement:\s*(.+)",
        r"Now,\s*write the number\s+(.+?)\s+in the Wonderland numeral system\.?",
        r"determine the falling distance for\s*(.+?)\s+given d\s*=",
    ]
    for pattern in patterns:
        m = re.search(pattern, prompt, re.I | re.S)
        if m:
            return m.group(1).strip().splitlines()[0].strip()
    return ""


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = text.strip("`")
    text = re.sub(r"\s+", " ", text)
    return text


def equivalent_answer(pred: Any, gold: Any, category: str = "") -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    if category == "bitwise":
        return p.lstrip("0") == g.lstrip("0")
    try:
        pf = float(p)
        gf = float(g)
        tolerance = 1.1e-2 if category in {"gravity", "unit"} else 1e-2
        return abs(pf - gf) <= tolerance
    except Exception:
        return False


def format_number(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text
