#!/usr/bin/env python3
# Single-file Kaggle solution for NVIDIA Nemotron Model Reasoning Challenge.

import gc
import json
import math
import os
import random
import re
import sys
import types
import zipfile
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

SEED = 42
random.seed(SEED)

KAGGLE_COMP_DIR = Path("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge")
WORK_DIR = Path("/kaggle/working")
MODEL_SLUG = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"

SYNTHETIC_PER_KIND = int(os.getenv("SYNTHETIC_PER_KIND", "3500"))
MAX_SYNTHETIC_RATIO = float(os.getenv("MAX_SYNTHETIC_RATIO", "0.30"))
VALIDATION_FRAC = float(os.getenv("VALIDATION_FRAC", "0.10"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))
EPOCHS = float(os.getenv("EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1.5e-4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "16"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))

ROMAN_MAP = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def install_local_mamba_shim() -> None:
    """Offline shim for the Nemotron remote code import path.

    Kaggle submissions often run without internet, while Nemotron-H remote code
    imports mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn at module import time.
    This shim provides a small PyTorch implementation for that import instead
    of trying to pip install mamba-ssm.
    """
    import torch

    def rmsnorm_fn(
        x,
        weight,
        bias=None,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=True,
        return_dropout_mask=False,
        **_,
    ):
        if residual is not None:
            x = x + residual.to(dtype=x.dtype)
        next_residual = x.float() if residual_in_fp32 else x
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        out = x.float() * torch.rsqrt(variance + eps)
        out = out.to(dtype=weight.dtype) * weight
        if bias is not None:
            out = out + bias
        out = out.to(dtype=x.dtype)

        if x1 is not None and weight1 is not None:
            variance1 = x1.float().pow(2).mean(dim=-1, keepdim=True)
            out1 = x1.float() * torch.rsqrt(variance1 + eps)
            out1 = out1.to(dtype=weight1.dtype) * weight1
            if bias1 is not None:
                out1 = out1 + bias1
            out = (out, out1.to(dtype=x1.dtype))

        if prenorm:
            result = (out, next_residual)
        else:
            result = out
        if return_dropout_mask:
            mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
            result = (*result, mask) if isinstance(result, tuple) else (result, mask)
        return result

    def selective_state_update(*_, **__):
        raise RuntimeError("selective_state_update is unavailable in the offline shim.")

    def mamba_split_conv1d_scan_combined(*_, **__):
        raise RuntimeError("mamba_split_conv1d_scan_combined is unavailable in the offline shim.")

    module_names = [
        "mamba_ssm",
        "mamba_ssm.ops",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.triton.selective_state_update",
        "mamba_ssm.ops.triton.ssd_combined",
    ]
    for name in module_names:
        module = sys.modules.setdefault(name, types.ModuleType(name))
        if name in {"mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"}:
            module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            module.__path__ = []
        else:
            module.__spec__ = ModuleSpec(name, loader=None)

    sys.modules["mamba_ssm"].__version__ = "2.0.4"

    sys.modules["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn = rmsnorm_fn
    sys.modules["mamba_ssm.ops.triton.selective_state_update"].selective_state_update = selective_state_update
    sys.modules["mamba_ssm.ops.triton.ssd_combined"].mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined
    sys.modules["mamba_ssm.ops.triton.ssd_combined"].mamba_chunk_scan_combined = mamba_split_conv1d_scan_combined

    try:
        import transformers.utils.import_utils as import_utils

        import_utils.is_mamba_2_ssm_available.cache_clear()
        import_utils.is_mamba_2_ssm_available = lambda: False
    except Exception as exc:
        log(f"Could not patch transformers mamba availability check: {exc}")


def find_csv(name: str) -> Path:
    candidates = [
        KAGGLE_COMP_DIR / name,
        Path("/kaggle/input") / name,
        Path.cwd() / name,
    ]
    candidates.extend(Path("/kaggle/input").rglob(name) if Path("/kaggle/input").exists() else [])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name}")


def clean_answer(value) -> str:
    return str(value).strip()


def roman(n: int) -> str:
    out = []
    for value, symbol in ROMAN_MAP:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def classify_equation_subtype(prompt: str) -> str:
    text = str(prompt)
    examples = re.findall(r"([^\n=]{1,40}?)\s*=\s*([^\n=]{1,40})", text)
    symbol_chars = set("@#$%^&*~!?<>|:;")
    has_symbols = any(any(ch in symbol_chars for ch in left + right) for left, right in examples)
    has_words = bool(re.search(r"\b[A-Za-z]+\s*[+\-*/=]\s*[A-Za-z]+", text))
    has_numeric_ops = bool(re.search(r"\d+\s*[/+*%\-]\s*\d+\s*=", text))
    if has_symbols or has_words:
        return "symbolic_cipher"
    if has_numeric_ops or "equation" in text.lower():
        return "numeric_equation"
    return "numeric_equation"


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
        return classify_equation_subtype(text)
    if "symbol transformation" in low or "symbol_transformation" in low:
        return classify_equation_subtype(text)
    return "unknown"


def make_training_text(prompt: str, answer: str, add_instruction: bool) -> Tuple[str, str]:
    if add_instruction:
        source = "Solve the puzzle. Output only the final answer.\n\n" + str(prompt).strip()
    else:
        source = str(prompt).strip()
    target = clean_answer(answer)
    return source, target


def generate_roman_rows(n_rows: int) -> List[Dict[str, str]]:
    rows = []
    for i in range(n_rows):
        query = random.randint(1, 3999)
        pool = [x for x in random.sample(range(1, 4000), random.randint(4, 6) + 1) if x != query]
        examples = pool[: random.randint(4, 6)]
        lines = [
            "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:"
        ]
        lines += [f"{x} -> {roman(x)}" for x in examples]
        lines.append(f"Now, write the number {query} in the Wonderland numeral system.")
        rows.append({
            "id": f"synthetic_roman_{i}",
            "prompt": "\n".join(lines),
            "answer": roman(query),
            "task_type": "numeral_system",
            "source": "synthetic_roman",
        })
    return rows


def generate_gravity_rows(n_rows: int) -> List[Dict[str, str]]:
    rows = []
    for i in range(n_rows):
        g = round(random.uniform(5.0, 15.0), 2)
        query_t = round(random.uniform(0.5, 5.0), 2)
        lines = [
            "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:"
        ]
        used = {query_t}
        for _ in range(random.randint(4, 7)):
            t = round(random.uniform(0.5, 5.0), 2)
            while t in used:
                t = round(random.uniform(0.5, 5.0), 2)
            used.add(t)
            d = 0.5 * g * t * t
            lines.append(f"For t = {t:.2f}s, distance = {d:.2f} m")
        lines.append(f"Now, determine the falling distance for t = {query_t:.2f}s given d = 0.5*g*t^2.")
        rows.append({
            "id": f"synthetic_gravity_{i}",
            "prompt": "\n".join(lines),
            "answer": f"{0.5 * g * query_t * query_t:.2f}",
            "task_type": "gravity",
            "source": "synthetic_gravity",
        })
    return rows


def generate_unit_rows(n_rows: int) -> List[Dict[str, str]]:
    rows = []
    for i in range(n_rows):
        ratio = round(random.uniform(0.2, 3.0), 4)
        query_x = round(random.uniform(1.0, 100.0), 2)
        lines = ["In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:"]
        used = {query_x}
        for _ in range(random.randint(4, 7)):
            x = round(random.uniform(1.0, 100.0), 2)
            while x in used:
                x = round(random.uniform(1.0, 100.0), 2)
            used.add(x)
            lines.append(f"{x:.2f} m becomes {x * ratio:.2f}")
        lines.append(f"Now, convert the following measurement: {query_x:.2f} m")
        rows.append({
            "id": f"synthetic_unit_{i}",
            "prompt": "\n".join(lines),
            "answer": f"{query_x * ratio:.2f}",
            "task_type": "unit_conversion",
            "source": "synthetic_unit_conversion",
        })
    return rows


BIT_OPS = [
    "identity", "not", "xor_const", "and_const", "or_const", "shift_left", "shift_right",
    "rotate_left", "rotate_right", "bit_reverse", "rotate_left_xor_const",
    "rotate_right_xor_const", "bit_reverse_xor_const", "not_rotate_left", "not_rotate_right",
]


def bits8(x: int) -> str:
    return format(x & 255, "08b")


def rotl(x: int, k: int) -> int:
    k %= 8
    return ((x << k) | (x >> (8 - k))) & 255


def rotr(x: int, k: int) -> int:
    k %= 8
    return ((x >> k) | (x << (8 - k))) & 255


def bit_reverse(x: int) -> int:
    return int(bits8(x)[::-1], 2)


def apply_bit_op(x: int, op: str, const: int, shift: int) -> int:
    if op == "identity":
        return x
    if op == "not":
        return (~x) & 255
    if op == "xor_const":
        return x ^ const
    if op == "and_const":
        return x & const
    if op == "or_const":
        return x | const
    if op == "shift_left":
        return (x << shift) & 255
    if op == "shift_right":
        return x >> shift
    if op == "rotate_left":
        return rotl(x, shift)
    if op == "rotate_right":
        return rotr(x, shift)
    if op == "bit_reverse":
        return bit_reverse(x)
    if op == "rotate_left_xor_const":
        return rotl(x, shift) ^ const
    if op == "rotate_right_xor_const":
        return rotr(x, shift) ^ const
    if op == "bit_reverse_xor_const":
        return bit_reverse(x) ^ const
    if op == "not_rotate_left":
        return rotl((~x) & 255, shift)
    if op == "not_rotate_right":
        return rotr((~x) & 255, shift)
    raise ValueError(op)


def generate_bit_rows(n_rows: int) -> List[Dict[str, str]]:
    rows = []
    for i in range(n_rows):
        op = random.choice(BIT_OPS)
        const = random.randint(0, 255)
        shift = random.randint(1, 7)
        query = random.randint(0, 255)
        seen = {query}
        lines = ["In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. Examples:"]
        for _ in range(random.randint(5, 8)):
            x = random.randint(0, 255)
            while x in seen:
                x = random.randint(0, 255)
            seen.add(x)
            lines.append(f"{bits8(x)} -> {bits8(apply_bit_op(x, op, const, shift))}")
        lines.append(f"Now, determine the output for: {bits8(query)}")
        rows.append({
            "id": f"synthetic_bit_{i}",
            "prompt": "\n".join(lines),
            "answer": bits8(apply_bit_op(query, op, const, shift)),
            "task_type": "bit_manipulation",
            "source": "synthetic_bit",
        })
    return rows


NUMERIC_OPS = [
    "add", "sub", "reverse_sub", "absdiff", "mul", "floor_div", "reverse_floor_div",
    "mod", "reverse_mod", "gcd", "lcm", "concat_lr", "concat_rl", "digitwise_add",
    "digitwise_absdiff", "reverse_result", "zero_padding",
]


def digitwise(a: int, b: int, fn) -> int:
    sa, sb = str(a), str(b)
    width = max(len(sa), len(sb))
    sa, sb = sa.zfill(width), sb.zfill(width)
    return int("".join(str(fn(int(x), int(y)) % 10) for x, y in zip(sa, sb)))


def apply_numeric_op(a: int, b: int, op: str) -> str:
    if op == "add":
        return str(a + b)
    if op == "sub":
        return str(a - b)
    if op == "reverse_sub":
        return str(b - a)
    if op == "absdiff":
        return str(abs(a - b))
    if op == "mul":
        return str(a * b)
    if op == "floor_div":
        return str(a // max(1, b))
    if op == "reverse_floor_div":
        return str(b // max(1, a))
    if op == "mod":
        return str(a % max(1, b))
    if op == "reverse_mod":
        return str(b % max(1, a))
    if op == "gcd":
        return str(math.gcd(a, b))
    if op == "lcm":
        return str(abs(a * b) // math.gcd(a, b))
    if op == "concat_lr":
        return f"{a}{b}"
    if op == "concat_rl":
        return f"{b}{a}"
    if op == "digitwise_add":
        return str(digitwise(a, b, lambda x, y: x + y))
    if op == "digitwise_absdiff":
        return str(digitwise(a, b, lambda x, y: abs(x - y)))
    if op == "reverse_result":
        return str(a + b)[::-1]
    if op == "zero_padding":
        return f"{a + b:04d}"
    raise ValueError(op)


def generate_numeric_rows(n_rows: int) -> List[Dict[str, str]]:
    rows = []
    for i in range(n_rows):
        op = random.choice(NUMERIC_OPS)
        query = (random.randint(1, 99), random.randint(1, 99))
        used = {query}
        lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. Examples:"]
        for _ in range(random.randint(5, 8)):
            a, b = random.randint(1, 99), random.randint(1, 99)
            while (a, b) in used:
                a, b = random.randint(1, 99), random.randint(1, 99)
            used.add((a, b))
            lines.append(f"{a}/{b} = {apply_numeric_op(a, b, op)}")
        qa, qb = query
        lines.append(f"Now, solve: {qa}/{qb} = ?")
        rows.append({
            "id": f"synthetic_numeric_{i}",
            "prompt": "\n".join(lines),
            "answer": apply_numeric_op(qa, qb, op),
            "task_type": "numeric_equation",
            "source": "synthetic_numeric_equation",
        })
    return rows


def generate_symbolic_rows_from_solver(max_rows: int) -> List[Dict[str, str]]:
    paths = list(Path("/kaggle/input").rglob("solver_results.parquet")) if Path("/kaggle/input").exists() else []
    paths += list(Path.cwd().rglob("solver_results.parquet"))
    if not paths:
        return []
    try:
        df = pd.read_parquet(paths[0])
    except Exception as exc:
        log(f"Could not read solver_results.parquet: {exc}")
        return []
    log(f"solver_results rows: {len(df)}")
    log(f"solver_results columns: {list(df.columns)}")
    if "solver_correct" in df.columns:
        df = df[df["solver_correct"] == True].copy()
    needed = {"id", "prompt", "answer"}
    if not needed.issubset(df.columns):
        return []
    rows = []
    for _, row in df.head(max_rows).iterrows():
        rows.append({
            "id": f"synthetic_symbolic_{row['id']}",
            "prompt": str(row["prompt"]),
            "answer": clean_answer(row["answer"]),
            "task_type": "symbolic_cipher",
            "source": "synthetic_symbolic_cipher",
        })
    return rows


def build_dataset(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    train["answer"] = train["answer"].map(clean_answer)
    train["task_type"] = train["prompt"].map(classify_prompt)
    train["source"] = "original"

    synthetic_rows = []
    synthetic_rows.extend(generate_roman_rows(SYNTHETIC_PER_KIND))
    synthetic_rows.extend(generate_gravity_rows(SYNTHETIC_PER_KIND))
    synthetic_rows.extend(generate_unit_rows(SYNTHETIC_PER_KIND))
    synthetic_rows.extend(generate_bit_rows(SYNTHETIC_PER_KIND))
    synthetic_rows.extend(generate_numeric_rows(SYNTHETIC_PER_KIND))
    synthetic_rows.extend(generate_symbolic_rows_from_solver(5000))

    synthetic = pd.DataFrame(synthetic_rows)
    if len(synthetic):
        max_synth = int(len(train) * MAX_SYNTHETIC_RATIO / max(1e-9, 1 - MAX_SYNTHETIC_RATIO))
        if len(synthetic) > max_synth:
            per_source = max(1, max_synth // synthetic["source"].nunique())
            parts = []
            for _, group in synthetic.groupby("source"):
                parts.append(group.sample(min(len(group), per_source), random_state=SEED))
            synthetic = pd.concat(parts, ignore_index=True)
        final = pd.concat([train[["id", "prompt", "answer", "task_type", "source"]], synthetic], ignore_index=True)
    else:
        final = train[["id", "prompt", "answer", "task_type", "source"]]

    final["answer"] = final["answer"].map(clean_answer)
    final = final[final["prompt"].astype(str).str.len() > 0]
    final = final[final["answer"].astype(str).str.len() > 0]
    final = final.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return final


def stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_parts, train_parts = [], []
    for _, group in df.groupby("task_type"):
        if len(group) < 2:
            train_parts.append(group)
            continue
        val_n = max(1, int(round(len(group) * VALIDATION_FRAC)))
        val = group.sample(n=val_n, random_state=SEED)
        train_part = group.drop(val.index)
        val_parts.append(val)
        train_parts.append(train_part)
    return (
        pd.concat(train_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True),
        pd.concat(val_parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True),
    )


class PromptAnswerDataset:
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        add_instruction = (idx % 10) >= 7
        prompt, answer = make_training_text(row["prompt"], row["answer"], add_instruction)
        prefix = prompt.rstrip() + "\n"
        full = prefix + answer + self.tokenizer.eos_token
        full_ids = self.tokenizer(full, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]
        prefix_ids = self.tokenizer(prefix, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]
        labels = full_ids.copy()
        prompt_len = min(len(prefix_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


@dataclass
class DataCollator:
    tokenizer: object

    def __call__(self, features):
        import torch

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def resolve_model_path() -> str:
    local_candidates = list(Path("/kaggle/input").glob("**/config.json")) if Path("/kaggle/input").exists() else []
    for cfg in local_candidates:
        parent = cfg.parent
        if "nemotron" in str(parent).lower():
            return str(parent)
    try:
        import kagglehub

        return kagglehub.model_download(MODEL_SLUG)
    except Exception as exc:
        log(f"kagglehub download failed, falling back to model slug: {exc}")
        return MODEL_SLUG


def available_lora_targets(model, preferred: Sequence[str]) -> List[str]:
    import torch

    suffixes = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffixes.add(name.split(".")[-1])
    targets = [x for x in preferred if x in suffixes]
    if targets:
        return targets
    fallback = sorted(suffixes)
    log(f"Preferred LoRA modules not found. Linear suffix fallback: {fallback[:20]}")
    return fallback


def train_lora(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Path:
    import inspect
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    install_local_mamba_shim()
    model_path = resolve_model_path()
    log(f"model_path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    targets = available_lora_targets(model, preferred)
    lora_config = LoraConfig(
        r=min(LORA_R, 32),
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = PromptAnswerDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = PromptAnswerDataset(val_df, tokenizer, MAX_LENGTH)
    args_kwargs = dict(
        output_dir=str(WORK_DIR / "trainer_output"),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        logging_steps=20,
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        args_kwargs["eval_strategy"] = "steps"
    else:
        args_kwargs["evaluation_strategy"] = "steps"
    args = TrainingArguments(**args_kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = WORK_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir


def package_submission(adapter_dir: Path) -> Path:
    submission_path = WORK_DIR / "submission.zip"
    if submission_path.exists():
        submission_path.unlink()
    allowed = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
    }
    with zipfile.ZipFile(submission_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for filename in allowed:
            path = adapter_dir / filename
            if path.exists():
                z.write(path, arcname=filename)
    with zipfile.ZipFile(submission_path, "r") as z:
        names = z.namelist()
    assert "adapter_config.json" in names, names
    assert "adapter_model.safetensors" in names, names
    log(f"submission.zip exists: {submission_path.exists()}")
    log(f"submission.zip size MB: {submission_path.stat().st_size / 1024 / 1024:.2f}")
    log("Files in submission.zip:")
    for name in names:
        log(f" - {name}")
    return submission_path


def print_report(train_raw: pd.DataFrame, test_raw: pd.DataFrame, full: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    log(f"train row count: {len(train_raw)}")
    log(f"test row count: {len(test_raw)}")
    log(f"train columns: {list(train_raw.columns)}")
    log(f"test columns: {list(test_raw.columns)}")
    log("task_type counts:")
    log(str(full["task_type"].value_counts(dropna=False)))
    log(f"unknown count: {(full['task_type'] == 'unknown').sum()}")
    log("synthetic counts by source:")
    log(str(full["source"].value_counts(dropna=False)))
    log(f"final train dataset size: {len(train_df)}")
    log(f"validation size: {len(val_df)}")
    log(f"LoRA r: {min(LORA_R, 32)}")
    log(f"LoRA alpha: {LORA_ALPHA}")


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    train_path = find_csv("train.csv")
    test_path = find_csv("test.csv")
    log(f"train_path: {train_path}")
    log(f"test_path: {test_path}")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    required_train = {"id", "prompt", "answer"}
    required_test = {"id", "prompt"}
    assert required_train.issubset(train_raw.columns), train_raw.columns
    assert required_test.issubset(test_raw.columns), test_raw.columns

    full = build_dataset(train_raw)
    train_df, val_df = stratified_split(full)
    print_report(train_raw, test_raw, full, train_df, val_df)

    adapter_dir = train_lora(train_df, val_df)
    submission_path = package_submission(adapter_dir)
    log(f"submission.zip path: {submission_path}")


if __name__ == "__main__":
    main()
