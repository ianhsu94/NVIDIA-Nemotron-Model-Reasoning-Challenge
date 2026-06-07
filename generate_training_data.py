#!/usr/bin/env python3
"""Generate high-score training data splits and augmentations for Nemotron."""

from __future__ import annotations

import json
import os
import random
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORIES = ["bitwise", "cipher", "numeral", "gravity", "unit", "symbol", "unknown"]
OUTPUT_COLUMNS = [
    "id",
    "prompt",
    "answer",
    "category",
    "subcategory",
    "source",
    "is_augmented",
    "parse_status",
    "target_text",
    "num_examples",
    "examples_json",
    "rule_type",
    "confidence_hint",
]

NOUNS = [
    "alice",
    "queen",
    "king",
    "dragon",
    "wizard",
    "hatter",
    "mouse",
    "rabbit",
    "princess",
    "student",
    "bird",
    "cat",
    "castle",
    "palace",
    "forest",
    "valley",
    "mirror",
    "mountain",
    "river",
    "garden",
    "door",
    "book",
    "secret",
    "wonderland",
]
VERBS = [
    "finds",
    "follows",
    "discovers",
    "creates",
    "reads",
    "writes",
    "chases",
    "watches",
    "imagines",
    "draws",
    "opens",
    "closes",
    "carries",
    "builds",
]
PREPOSITIONS = ["near", "inside", "under", "behind", "beside", "around", "through", "above", "below"]
ADJECTIVES = ["golden", "wise", "magical", "mysterious", "hidden", "silent", "bright", "ancient"]
SYMBOLS = list("!@#$%^&*()[]{}<>?/\\|`~'\"")


def find_train_csv() -> Path:
    env_path = os.getenv("TRAIN_CSV_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"TRAIN_CSV_PATH does not exist: {path}")

    candidates = [
        Path("/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv"),
        Path("/mnt/data/nemotron/train.csv"),
        Path("/mnt/data/train.csv"),
        Path.cwd() / "train.csv",
    ]
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").glob("*/train.csv"))

    for path in candidates:
        if path.exists():
            return path

    fallback = Path.cwd() / "nemotron_data_ready" / "train_ready_v1.csv"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Could not find train.csv. Searched /kaggle/input/**/train.csv, /mnt/data, cwd, "
        "and fallback nemotron_data_ready/train_ready_v1.csv."
    )


def get_output_dir() -> Path:
    out_dir = Path("/kaggle/working/training_data") if Path("/kaggle/working").exists() else Path.cwd() / "training_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_input_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if path.name == "train_ready_v1.csv" and "source" in df.columns:
        df = df[df["source"].astype(str).eq("original")].copy()
    required = {"id", "prompt", "answer"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input data missing columns: {missing}")
    df = df[["id", "prompt", "answer"]].copy()
    for col in ["id", "prompt", "answer"]:
        df[col] = df[col].astype(str)
    return df.reset_index(drop=True)


def classify_prompt(prompt: str) -> tuple[str, str]:
    p = prompt.lower()
    if any(x in p for x in ["bit manipulation", "8-bit binary", "bit shifts", "rotations", " xor ", " and ", " or ", " not "]):
        return "bitwise", "bitwise_rule_discovery"
    if any(x in p for x in ["secret encryption rules are used on text", "decrypt the following text", "encryption", "decrypt", "cipher"]):
        return "cipher", "substitution_or_word_cipher"
    if any(x in p for x in ["numeral system", "wonderland numeral system", "roman", "write the number", "different numeral system"]):
        if re.search(r"\b[ivxlcdm]{2,}\b", p):
            return "numeral", "roman_to_decimal"
        return "numeral", "decimal_to_roman"
    if any(x in p for x in ["falling distance", "0.5*g*t^2", "free fall", "gravity"]) or re.search(r"\bg\s*=", p):
        if any(x in p for x in ["doubled", "changed", "scaled"]):
            return "gravity", "gravity_scaling"
        return "gravity", "free_fall_formula"
    if any(x in p for x in ["convert the following measurement", "measurement", " unit "]):
        if any(x in p for x in [" km", " cm", " mm", " meter", " m "]):
            return "unit", "length_unit"
        if any(x in p for x in [" kg", " mg", " gram", " mass"]):
            return "unit", "mass_unit"
        if any(x in p for x in [" hour", " min", " second", " s "]):
            return "unit", "time_unit"
        return "unit", "unknown_unit_conversion"
    if any(x in p for x in ["secret set of transformation rules", "transformation rules", "symbols", "equations", "determine the result for"]):
        return "symbol", "alice_symbol_equation"
    return "unknown", "unknown"


def extract_examples_and_target(prompt: str, category: str) -> dict[str, Any]:
    examples: list[tuple[str, str]] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("now,"):
            continue
        arrow = re.match(r"^(.+?)\s*->\s*(.+?)$", line)
        equation = re.match(r"^(.+?)\s*=\s*(.+?)$", line)
        unit_match = re.match(r"^(.+?)\s+becomes\s+(.+?)$", line, flags=re.IGNORECASE)
        gravity_match = re.match(r"^For\s+(.+?),\s*distance\s*=\s*(.+?)$", line, flags=re.IGNORECASE)
        match = arrow or unit_match or gravity_match or (equation if category == "symbol" else None)
        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()
            if left and right and len(left) <= 300 and len(right) <= 300:
                examples.append((left, right))

    target_patterns = {
        "bitwise": r"Now,\s*determine the output for:\s*(.+)",
        "cipher": r"Now,\s*decrypt the following text:\s*(.+)",
        "numeral": r"Now,\s*write the number\s+(.+?)\s+in the Wonderland numeral system\.?",
        "gravity": r"determine the falling distance for\s*(.+?)\s+given d\s*=\s*0\.5\*g\*t\^2\.?",
        "unit": r"convert the following measurement:\s*(.+)",
        "symbol": r"Now,\s*determine the result for:\s*(.+)",
    }
    target_text = ""
    pattern = target_patterns.get(category)
    if pattern:
        match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
        if match:
            target_text = match.group(1).strip().splitlines()[0].strip()

    parse_status = "ok" if target_text and examples else "partial" if target_text or examples else "failed"
    return {
        "examples": examples,
        "target_text": target_text,
        "num_examples": len(examples),
        "parse_status": parse_status,
    }


def build_classified_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        category, subcategory = classify_prompt(row["prompt"])
        parsed = extract_examples_and_target(row["prompt"], category)
        rows.append(
            {
                "id": str(row["id"]),
                "prompt": str(row["prompt"]),
                "answer": str(row["answer"]).strip(),
                "category": category,
                "subcategory": subcategory,
                "source": "original",
                "is_augmented": False,
                "parse_status": parsed["parse_status"],
                "target_text": parsed["target_text"],
                "num_examples": parsed["num_examples"],
                "examples_json": json.dumps(parsed["examples"], ensure_ascii=False),
                "rule_type": subcategory,
                "confidence_hint": "original_classified",
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def random_phrase() -> str:
    size = random.randint(3, 5)
    words = []
    for _ in range(size):
        bucket = random.random()
        if bucket < 0.45:
            words.append(random.choice(NOUNS))
        elif bucket < 0.70:
            words.append(random.choice(VERBS))
        elif bucket < 0.85:
            words.append(random.choice(PREPOSITIONS))
        else:
            words.append(random.choice(ADJECTIVES))
    return " ".join(words)


def caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch in string.ascii_lowercase:
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)


def atbash(text: str) -> str:
    return "".join(chr(122 - (ord(ch) - 97)) if ch in string.ascii_lowercase else ch for ch in text)


def vowel_shift(text: str, shift: int) -> str:
    vowels = "aeiou"
    return "".join(vowels[(vowels.index(ch) + shift) % 5] if ch in vowels else ch for ch in text)


def mono_mapping() -> dict[str, str]:
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    random.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def apply_mono(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def cipher_rule() -> tuple[str, Any, Any]:
    rule = random.choice(
        ["caesar_shift", "rot13", "atbash", "reverse_word", "reverse_phrase", "mono_substitution", "vowel_shift", "multi_step_reverse_caesar"]
    )
    if rule == "caesar_shift":
        shift = random.choice([1, 2, 3, 4, 5, 7, 9, 11, 15, 19, 23])
        return rule, lambda plain: caesar(plain, shift), {"shift": shift}
    if rule == "rot13":
        return rule, lambda plain: caesar(plain, 13), {}
    if rule == "atbash":
        return rule, atbash, {}
    if rule == "reverse_word":
        return rule, lambda plain: " ".join(word[::-1] for word in plain.split()), {}
    if rule == "reverse_phrase":
        return rule, lambda plain: " ".join(reversed(plain.split())), {}
    if rule == "mono_substitution":
        mapping = mono_mapping()
        return rule, lambda plain: apply_mono(plain, mapping), {"mapping": mapping}
    if rule == "vowel_shift":
        shift = random.choice([1, 2, 3, 4])
        return rule, lambda plain: vowel_shift(plain, shift), {"shift": shift}
    shift = random.choice([2, 3, 5, 8, 13])
    return rule, lambda plain: caesar(" ".join(reversed(plain.split())), shift), {"shift": shift}


def augment_cipher(cipher_df: pd.DataFrame, target_multiplier: int = 4) -> pd.DataFrame:
    n_original = len(cipher_df)
    target_count = max(3000, n_original * target_multiplier)
    rows = []
    for i in range(target_count):
        rule_type, encode, details = cipher_rule()
        plain_examples = []
        while len(plain_examples) < random.randint(4, 7):
            phrase = random_phrase()
            if phrase not in plain_examples:
                plain_examples.append(phrase)
        target_plain = random_phrase()
        while target_plain in plain_examples:
            target_plain = random_phrase()

        examples = [(encode(plain), plain) for plain in plain_examples]
        target_cipher = encode(target_plain)
        prompt_lines = [
            "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:",
            *[f"{cipher} -> {plain}" for cipher, plain in examples],
            f"Now, decrypt the following text: {target_cipher}",
        ]
        rows.append(
            {
                "id": f"aug_cipher_{i:06d}",
                "prompt": "\n".join(prompt_lines),
                "answer": target_plain,
                "category": "cipher",
                "subcategory": rule_type,
                "source": "augmented",
                "is_augmented": True,
                "parse_status": "ok",
                "target_text": target_cipher,
                "num_examples": len(examples),
                "examples_json": json.dumps(examples, ensure_ascii=False),
                "rule_type": rule_type,
                "confidence_hint": json.dumps(details, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def random_symbol_string(min_len: int = 3, max_len: int = 7) -> str:
    return "".join(random.choice(SYMBOLS) for _ in range(random.randint(min_len, max_len)))


def make_symbol_transform() -> tuple[str, Any, dict[str, Any]]:
    rule = random.choice(
        [
            "direct_symbol_map",
            "position_map",
            "delete_rule",
            "keep_only_rule",
            "reorder_reverse",
            "reorder_rotate_left",
            "reorder_rotate_right",
            "pair_interaction",
            "operator_rule",
            "multi_step_symbol",
            "alice_symbol_equation",
        ]
    )
    if rule == "direct_symbol_map":
        src = random.sample(SYMBOLS, k=min(12, len(SYMBOLS)))
        dst = random.sample(SYMBOLS, k=min(12, len(SYMBOLS)))
        mapping = dict(zip(src, dst))
        return rule, lambda s: "".join(mapping.get(ch, ch) for ch in s), {"mapping": mapping}
    if rule == "position_map":
        mask = random.choice(["even", "odd"])
        repl = random.choice(SYMBOLS)
        return rule, lambda s: "".join(repl if (i % 2 == (0 if mask == "even" else 1)) else ch for i, ch in enumerate(s)), {"mask": mask, "replace": repl}
    if rule == "delete_rule":
        delete = set(random.sample(SYMBOLS, k=3))
        return rule, lambda s: "".join(ch for ch in s if ch not in delete) or random.choice(SYMBOLS), {"delete": sorted(delete)}
    if rule == "keep_only_rule":
        keep = set(random.sample(SYMBOLS, k=5))
        return rule, lambda s: "".join(ch for ch in s if ch in keep) or random.choice(sorted(keep)), {"keep": sorted(keep)}
    if rule == "reorder_reverse":
        return rule, lambda s: s[::-1], {}
    if rule == "reorder_rotate_left":
        amount = random.randint(1, 3)
        return rule, lambda s: s[amount % len(s) :] + s[: amount % len(s)], {"amount": amount}
    if rule == "reorder_rotate_right":
        amount = random.randint(1, 3)
        return rule, lambda s: s[-(amount % len(s)) :] + s[: -(amount % len(s))], {"amount": amount}
    if rule == "pair_interaction":
        marker = random.choice(SYMBOLS)
        return rule, lambda s: "".join((a + marker) for a in s[::2])[: max(1, len(s))], {"marker": marker}
    if rule == "operator_rule":
        op = random.choice(["first_last", "duplicate_first", "drop_middle"])
        def transform(s: str) -> str:
            if op == "first_last":
                return s[0] + s[-1]
            if op == "duplicate_first":
                return s[0] + s
            mid = len(s) // 2
            return s[:mid] + s[mid + 1 :]
        return rule, transform, {"op": op}
    if rule == "multi_step_symbol":
        mapping = dict(zip(random.sample(SYMBOLS, 8), random.sample(SYMBOLS, 8)))
        return rule, lambda s: "".join(mapping.get(ch, ch) for ch in s)[::-1], {"mapping": mapping, "then": "reverse"}
    shift = random.randint(1, 5)
    return rule, lambda s: "".join(SYMBOLS[(SYMBOLS.index(ch) + shift) % len(SYMBOLS)] for ch in s), {"shift": shift}


def augment_symbol(symbol_df: pd.DataFrame, target_multiplier: int = 6) -> pd.DataFrame:
    n_original = len(symbol_df)
    target_count = max(6000, n_original * target_multiplier)
    rows = []
    for i in range(target_count):
        rule_type, transform, details = make_symbol_transform()
        input_examples = []
        while len(input_examples) < random.randint(4, 8):
            value = random_symbol_string()
            if value not in input_examples:
                input_examples.append(value)
        target_input = random_symbol_string()
        while target_input in input_examples:
            target_input = random_symbol_string()

        examples = [(value, transform(value)) for value in input_examples]
        target_output = transform(target_input)
        prompt_lines = [
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
            "Below are a few examples:",
            "",
            *[f"{src} = {dst}" for src, dst in examples],
            "",
            f"Now, determine the result for: {target_input}",
        ]
        rows.append(
            {
                "id": f"aug_symbol_{i:06d}",
                "prompt": "\n".join(prompt_lines),
                "answer": target_output,
                "category": "symbol",
                "subcategory": rule_type,
                "source": "augmented",
                "is_augmented": True,
                "parse_status": "ok",
                "target_text": target_input,
                "num_examples": len(examples),
                "examples_json": json.dumps(examples, ensure_ascii=False),
                "rule_type": rule_type,
                "confidence_hint": json.dumps(details, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def split_and_save_categories(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = {
        "numeral": "numeral_train.csv",
        "gravity": "gravity_train.csv",
        "unit": "unit_train.csv",
        "bitwise": "bitwise_train.csv",
        "cipher": "cipher_train.csv",
        "symbol": "symbol_train.csv",
    }
    splits = {}
    for category, filename in mapping.items():
        part = df[df["category"].eq(category)].copy()
        splits[category] = part
        save_csv(part, out_dir / filename)
    unknown = df[df["category"].eq("unknown")].copy()
    if len(unknown):
        save_csv(unknown, out_dir / "unknown_train.csv")
    return splits


def save_statistics(out_dir: Path, classified: pd.DataFrame, cipher_aug: pd.DataFrame, symbol_aug: pd.DataFrame) -> dict[str, Any]:
    original_counts = {cat: int((classified["category"] == cat).sum()) for cat in CATEGORIES}
    stats = {
        "total_original": int(len(classified)),
        "original_counts": original_counts,
        "augmented_counts": {
            "cipher_augmented": int(len(cipher_aug)),
            "symbol_augmented": int(len(symbol_aug)),
        },
        "full_counts": {
            "cipher_full": int(original_counts["cipher"] + len(cipher_aug)),
            "symbol_full": int(original_counts["symbol"] + len(symbol_aug)),
        },
    }
    with (out_dir / "category_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    rows = []
    for category, count in original_counts.items():
        rows.append({"section": "original", "name": category, "count": count})
    rows.extend(
        [
            {"section": "augmented", "name": "cipher_augmented", "count": len(cipher_aug)},
            {"section": "augmented", "name": "symbol_augmented", "count": len(symbol_aug)},
            {"section": "full", "name": "cipher_full", "count": stats["full_counts"]["cipher_full"]},
            {"section": "full", "name": "symbol_full", "count": stats["full_counts"]["symbol_full"]},
        ]
    )
    pd.DataFrame(rows).to_csv(out_dir / "category_statistics.csv", index=False)
    return stats


def save_samples_markdown(out_dir: Path, classified: pd.DataFrame, cipher_aug: pd.DataFrame, symbol_aug: pd.DataFrame) -> None:
    combined = pd.concat([classified, cipher_aug, symbol_aug], ignore_index=True)
    lines = ["# Category Samples", ""]
    for category in ["bitwise", "cipher", "numeral", "gravity", "unit", "symbol"]:
        lines.extend([f"## {category}", ""])
        samples = combined[combined["category"].eq(category)].head(5)
        for i, row in enumerate(samples.itertuples(index=False), 1):
            lines.extend(
                [
                    f"### sample {i}",
                    f"- id: {row.id}",
                    f"- subcategory: {row.subcategory}",
                    f"- target_text: {row.target_text}",
                    f"- answer: {row.answer}",
                    "",
                    "```text",
                    str(row.prompt),
                    "```",
                    "",
                ]
            )
    (out_dir / "category_samples.md").write_text("\n".join(lines), encoding="utf-8")


def assert_outputs(out_dir: Path) -> None:
    required_files = [
        "numeral_train.csv",
        "gravity_train.csv",
        "unit_train.csv",
        "bitwise_train.csv",
        "cipher_train.csv",
        "symbol_train.csv",
        "cipher_augmented.csv",
        "symbol_augmented.csv",
        "cipher_full.csv",
        "symbol_full.csv",
        "all_classified_train.csv",
        "category_statistics.json",
        "category_statistics.csv",
        "category_samples.md",
    ]
    missing = [name for name in required_files if not (out_dir / name).exists()]
    if missing:
        raise AssertionError(f"Missing output files: {missing}")


def main() -> None:
    train_path = find_train_csv()
    out_dir = get_output_dir()

    input_df = load_input_data(train_path)
    classified = build_classified_dataframe(input_df)
    if int((classified["category"] == "unknown").sum()) != 0:
        unknown_sample = classified[classified["category"].eq("unknown")][["id", "prompt"]].head(20)
        raise RuntimeError(f"Classification produced unknown rows:\n{unknown_sample}")

    splits = split_and_save_categories(classified, out_dir)
    cipher_aug = augment_cipher(splits["cipher"], target_multiplier=4)
    symbol_aug = augment_symbol(splits["symbol"], target_multiplier=6)
    cipher_full = pd.concat([splits["cipher"], cipher_aug], ignore_index=True)
    symbol_full = pd.concat([splits["symbol"], symbol_aug], ignore_index=True)

    save_csv(classified, out_dir / "all_classified_train.csv")
    save_csv(cipher_aug, out_dir / "cipher_augmented.csv")
    save_csv(symbol_aug, out_dir / "symbol_augmented.csv")
    save_csv(cipher_full, out_dir / "cipher_full.csv")
    save_csv(symbol_full, out_dir / "symbol_full.csv")
    stats = save_statistics(out_dir, classified, cipher_aug, symbol_aug)
    save_samples_markdown(out_dir, classified, cipher_aug, symbol_aug)
    assert_outputs(out_dir)

    print("=== Training Data Generation Complete ===")
    print(f"Input train.csv: {train_path}")
    print(f"Output dir: {out_dir}")
    print("")
    print("Original category counts:")
    for category in sorted(stats["original_counts"]):
        print(f"{category}: {stats['original_counts'][category]}")
    print("")
    print("Augmented counts:")
    for name, count in stats["augmented_counts"].items():
        print(f"{name}: {count}")
    print("")
    print("Full counts:")
    for name, count in stats["full_counts"].items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
