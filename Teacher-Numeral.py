#!/usr/bin/env python3
"""Train Teacher-Numeral LoRA adapter for the Nemotron Reasoning Challenge."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEACHER_NAME = "Teacher-Numeral"
TEACHER_TYPE = "numeral"
TARGET_TASK_TYPE = "numeral_system"
TARGET_SOURCES = {"original", "original_replay", "synthetic_roman"}
ADAPTER_BASENAME = "adapter_teacher_numeral"
REPORT_BASENAME = "report_teacher_numeral.txt"
SANITY_MAX_NEW_TOKENS = 32

TEACHER_TARGETS = [
    "teacher_general",
    "teacher_numeral",
    "teacher_gravity",
    "teacher_unit",
    "teacher_symbolic_solver",
    "teacher_symbolic_general",
    "teacher_text_mapping",
    "teacher_text_format",
    "teacher_numeric_arithmetic",
    "teacher_numeric_digitstring",
    "teacher_numeric_mixed",
    "teacher_bit_original",
    "teacher_bit_operation",
    "teacher_bit_augmented",
]
TEACHER_ALIASES = {
    "general": "teacher_general",
    "numeral": "teacher_numeral",
    "gravity": "teacher_gravity",
    "unit": "teacher_unit",
}
ALLOWED_TEACHERS = list(TEACHER_ALIASES) + TEACHER_TARGETS
TEACHER_MAIN_TASK_MAP = {
    "teacher_general": [
        "bit_manipulation",
        "text_decryption",
        "numeral_system",
        "unit_conversion",
        "gravity",
        "symbolic_cipher",
        "numeric_equation",
    ],
    "teacher_numeral": ["numeral_system"],
    "teacher_gravity": ["gravity"],
    "teacher_unit": ["unit_conversion"],
    "teacher_symbolic_solver": ["symbolic_cipher"],
    "teacher_symbolic_general": ["symbolic_cipher"],
    "teacher_text_mapping": ["text_decryption"],
    "teacher_text_format": ["text_decryption"],
    "teacher_numeric_arithmetic": ["numeric_equation"],
    "teacher_numeric_digitstring": ["numeric_equation"],
    "teacher_numeric_mixed": ["numeric_equation"],
    "teacher_bit_original": ["bit_manipulation"],
    "teacher_bit_operation": ["bit_manipulation"],
    "teacher_bit_augmented": ["bit_manipulation"],
}
TEACHER_SANITY_TASK_MAP = {
    teacher: task_types[0] for teacher, task_types in TEACHER_MAIN_TASK_MAP.items()
}
TEACHER_MAX_NEW_TOKENS_MAP = {
    "teacher_general": 64,
    "teacher_numeral": 32,
    "teacher_gravity": 16,
    "teacher_unit": 16,
    "teacher_symbolic_solver": 64,
    "teacher_symbolic_general": 64,
    "teacher_text_mapping": 64,
    "teacher_text_format": 64,
    "teacher_numeric_arithmetic": 32,
    "teacher_numeric_digitstring": 32,
    "teacher_numeric_mixed": 32,
    "teacher_bit_original": 32,
    "teacher_bit_operation": 32,
    "teacher_bit_augmented": 32,
}

MODEL_SLUG = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
SEED = 42
MAX_LENGTH_DEFAULT = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_PREFERENCE = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
REQUIRED_COLUMNS = [
    "id",
    "prompt",
    "answer",
    "task_type",
    "source",
    "is_replay",
    "teacher_target",
    "teacher_data_role",
]
BAD_ANSWER_PATTERNS = ["Answer:", "The answer is", "Let's solve", "step by step", "```"]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Easy Teacher LoRA adapter.")
    parser.add_argument("--teacher", type=str, default=TEACHER_TYPE, choices=ALLOWED_TEACHERS)
    parser.add_argument("--data", type=str, default="", help="Path to teacher training parquet.")
    parser.add_argument("--model", type=str, default="", help="Local HuggingFace model folder.")
    parser.add_argument("--output-root", type=str, default="", help="Output root. Defaults to /kaggle/working or cwd.")
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", MAX_LENGTH_DEFAULT)))
    parser.add_argument("--epochs", type=float, default=float(os.getenv("NUM_EPOCHS", "1")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("LEARNING_RATE", "1e-4")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "2")))
    parser.add_argument("--grad-accum", type=int, default=int(os.getenv("GRAD_ACCUM", "8")))
    parser.add_argument("--warmup-ratio", type=float, default=float(os.getenv("WARMUP_RATIO", "0.03")))
    parser.add_argument("--tokenize-num-proc", type=int, default=int(os.getenv("TOKENIZE_NUM_PROC", "2")))
    parser.add_argument("--sanity-n", type=int, default=int(os.getenv("SANITY_N", "20")))
    return parser.parse_known_args()[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root)
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()


def resolve_teacher_config(teacher_type: str) -> dict[str, Any]:
    if teacher_type not in ALLOWED_TEACHERS:
        raise ValueError(f"Invalid teacher {teacher_type!r}. Allowed: {ALLOWED_TEACHERS}")
    teacher_target = TEACHER_ALIASES.get(teacher_type, teacher_type)
    label = teacher_target.removeprefix("teacher_")
    display = "Teacher-" + "-".join(part.capitalize() for part in label.split("_"))
    return {
        "teacher_type": teacher_type,
        "teacher_name": display,
        "required_filename": f"{teacher_target}_train.parquet",
        "teacher_target": teacher_target,
        "main_task_types": TEACHER_MAIN_TASK_MAP[teacher_target],
        "main_task_type": TEACHER_SANITY_TASK_MAP[teacher_target],
        "adapter_basename": f"adapter_{teacher_target}",
        "report_basename": f"report_{teacher_target}.txt",
        "sanity_max_new_tokens": TEACHER_MAX_NEW_TOKENS_MAP[teacher_target],
    }


def apply_teacher_config(config: dict[str, Any]) -> None:
    global TEACHER_NAME, TEACHER_TYPE, TARGET_TASK_TYPE, ADAPTER_BASENAME
    global REPORT_BASENAME, SANITY_MAX_NEW_TOKENS

    TEACHER_NAME = config["teacher_name"]
    TEACHER_TYPE = config["teacher_type"]
    TARGET_TASK_TYPE = config["main_task_type"]
    ADAPTER_BASENAME = config["adapter_basename"]
    REPORT_BASENAME = config["report_basename"]
    SANITY_MAX_NEW_TOKENS = config["sanity_max_new_tokens"]


def find_teacher_train_file(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.data:
        path = Path(args.data)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    required_filename = config["required_filename"]
    searched = [
        f"/kaggle/input/**/{required_filename}",
        f"/kaggle/working/easy_teacher_data/**/{required_filename}",
        f"{Path.cwd() / 'easy_teacher_data'}/**/{required_filename}",
        str(Path.cwd() / required_filename),
    ]
    candidates = [Path.cwd() / required_filename]
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").rglob(required_filename))
    if Path("/kaggle/working/easy_teacher_data").exists():
        candidates.extend(Path("/kaggle/working/easy_teacher_data").rglob(required_filename))
    if (Path.cwd() / "easy_teacher_data").exists():
        candidates.extend((Path.cwd() / "easy_teacher_data").rglob(required_filename))

    seen = set()
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").rglob(required_filename))
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.name == required_filename:
            return path
    raise FileNotFoundError(
        "Cannot find teacher training file.\n"
        f"TEACHER_TYPE: {config['teacher_type']}\n"
        f"Required file: {required_filename}\n"
        "Searched:\n- " + "\n- ".join(searched)
    )


def load_teacher_train_data(path: Path):
    import pandas as pd

    df = pd.read_parquet(path)
    for col in ["id", "prompt", "answer", "task_type", "source", "teacher_target", "teacher_data_role"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def validate_teacher_train_data(df, teacher_type: str, config: dict[str, Any]) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["id", "prompt", "answer", "task_type", "source", "teacher_target", "teacher_data_role"]:
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if bad.any():
            sample = df.loc[bad, ["id", "source", col]].head(20)
            raise ValueError(f"Empty values found in {col}:\n{sample}")
    if df["id"].duplicated().any():
        sample = df.loc[df["id"].duplicated(keep=False), ["id", "task_type", "source"]].head(20)
        raise ValueError(f"Duplicate id values found:\n{sample}")
    for pattern in BAD_ANSWER_PATTERNS:
        bad = df["answer"].astype(str).str.contains(re.escape(pattern), case=False, na=False)
        if bad.any():
            sample = df.loc[bad, ["id", "task_type", "source", "answer"]].head(10)
            raise ValueError(f"Answer contains forbidden pattern {pattern!r}:\n{sample}")
    expected_teacher_target = config["teacher_target"]
    wrong_target = df["teacher_target"] != expected_teacher_target
    if wrong_target.any():
        sample = df.loc[wrong_target, ["id", "teacher_target"]].head(20)
        raise ValueError(f"teacher_target must all be {expected_teacher_target!r}:\n{sample}")

    allowed_roles = {"main", "format_replay", "strategy_replay"}
    bad_roles = sorted(set(df["teacher_data_role"]) - allowed_roles)
    if bad_roles:
        raise ValueError(f"Invalid teacher_data_role values: {bad_roles}")

    main = df["teacher_data_role"] == "main"
    replay = df["teacher_data_role"].isin(["format_replay", "strategy_replay"])
    main_task_types = set(config["main_task_types"])
    wrong_main = main & ~df["task_type"].isin(main_task_types)
    if wrong_main.any():
        sample = df.loc[wrong_main, ["id", "task_type", "teacher_data_role"]].head(20)
        raise ValueError(f"main rows must use task_type in {sorted(main_task_types)!r}:\n{sample}")
    wrong_replay = replay & (df["task_type"].isin(main_task_types) | (df["source"] != "original"))
    if wrong_replay.any():
        sample = df.loc[wrong_replay, ["id", "task_type", "source", "teacher_data_role"]].head(20)
        raise ValueError(f"replay rows must be source original and non-main task:\n{sample}")
    if len(df) < 2000:
        raise ValueError(f"Teacher training rows too small: {len(df)}")
    if int(main.sum()) < 1000:
        raise ValueError(f"main rows too small: {int(main.sum())}")


def make_user_prompt(prompt: str, use_instruction: bool) -> str:
    prompt = str(prompt).strip()
    if use_instruction:
        return "Solve the puzzle. Output only the final answer.\n\n" + prompt
    return prompt


def render_pair(tokenizer, prompt: str, answer: str, use_instruction: bool) -> dict[str, str]:
    user_content = make_user_prompt(prompt, use_instruction)
    answer = str(answer).strip()
    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}, {"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = f"User:\n{user_content}\n\nAssistant:\n"
        full_text = prompt_text + answer + (tokenizer.eos_token or "")
    return {"prompt_text": prompt_text, "full_text": full_text}


def build_training_texts(df, tokenizer, teacher_type: str):
    from datasets import Dataset

    rng = random.Random(SEED)
    instruction_ids = set(rng.sample(range(len(df)), k=round(len(df) * 0.30)))
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        pair = render_pair(tokenizer, row["prompt"], row["answer"], idx in instruction_ids)
        rows.append(
            {
                "id": str(row["id"]),
                "task_type": str(row["task_type"]),
                "source": str(row["source"]),
                "answer": str(row["answer"]).strip(),
                **pair,
            }
        )
    return Dataset.from_list(rows)


def compute_token_length_stats(dataset, tokenizer) -> dict[str, Any]:
    import pandas as pd

    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in dataset["full_text"]]
    series = pd.Series(lengths)
    return {
        "token_length_p50": float(series.quantile(0.50)),
        "token_length_p90": float(series.quantile(0.90)),
        "token_length_p95": float(series.quantile(0.95)),
        "token_length_p99": float(series.quantile(0.99)),
        "token_length_max": int(series.max()),
    }


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model:
        path = Path(args.model)
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)
    if Path("/kaggle/input").exists():
        model_candidates = []
        for config in Path("/kaggle/input").rglob("config.json"):
            folder = config.parent
            folder_text = str(folder).lower()
            has_tokenizer = (
                (folder / "tokenizer.json").exists()
                or (folder / "tokenizer.model").exists()
                or any(path.name.startswith("tokenizer") for path in folder.iterdir() if path.is_file())
            )
            if "nemotron" in folder_text:
                model_candidates.append((folder, has_tokenizer))
        tokenizer_candidates = [folder for folder, has_tokenizer in model_candidates if has_tokenizer]
        if tokenizer_candidates:
            return str(tokenizer_candidates[0])
        if model_candidates:
            return str(model_candidates[0][0])
    try:
        import kagglehub

        return kagglehub.model_download(MODEL_SLUG)
    except Exception as exc:
        raise FileNotFoundError(f"Could not resolve Nemotron model. Attach {MODEL_SLUG}. Original error: {exc}") from exc


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log(f"torch version: {torch.__version__}")
    log(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for teacher training.")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        log(f"GPU {idx} name: {props.name}")
        log(f"GPU {idx} total memory GB: {props.total_memory / 1024 ** 3:.2f}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log(f"training dtype: {dtype}")

    model_path = resolve_model_path(args)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    max_memory = {idx: "12GiB" for idx in range(torch.cuda.device_count())}
    max_memory["cpu"] = "48GiB"
    offload_dir = output_root(args) / "model_offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    log(f"max_memory: {max_memory}")
    log(f"offload_folder: {offload_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model_path, tokenizer, model


def find_lora_target_modules(model) -> list[str]:
    import torch

    suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    targets = [name for name in LORA_TARGET_PREFERENCE if name in suffixes]
    if not targets:
        targets = sorted(suffixes)
    if not targets:
        raise ValueError("No torch.nn.Linear modules found for LoRA.")
    return targets


def apply_lora(model, target_modules: list[str]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def tokenize_dataset(dataset, tokenizer, max_length: int, num_proc: int):
    def tokenize_batch(batch):
        full = tokenizer(batch["full_text"], add_special_tokens=False, truncation=True, max_length=max_length)
        prompt = tokenizer(batch["prompt_text"], add_special_tokens=False, truncation=True, max_length=max_length)
        labels = []
        for input_ids, prompt_ids in zip(full["input_ids"], prompt["input_ids"]):
            row_labels = list(input_ids)
            prompt_len = min(len(prompt_ids), len(row_labels))
            row_labels[:prompt_len] = [-100] * prompt_len
            if all(label == -100 for label in row_labels) and row_labels:
                row_labels[-1] = input_ids[-1]
            labels.append(row_labels)
        full["labels"] = labels
        return full

    return dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=max(1, num_proc),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {TEACHER_NAME}",
    )


@dataclass
class AnswerOnlyCollator:
    tokenizer: Any

    def __call__(self, features):
        import torch

        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def train_teacher(model, tokenizer, tokenized_dataset, args: argparse.Namespace, root: Path):
    from transformers import Trainer, TrainingArguments

    trainer_dir = root / f"trainer_{ADAPTER_BASENAME}"
    kwargs = dict(
        output_dir=str(trainer_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    import torch
    if torch.cuda.is_bf16_supported():
        kwargs["bf16"] = True
    else:
        kwargs["fp16"] = True
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=tokenized_dataset,
        data_collator=AnswerOnlyCollator(tokenizer),
    )
    trainer.train()
    return model


def run_sanity_generation(model, tokenizer, teacher_df, args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    if args.sanity_n <= 0:
        return []
    sample = teacher_df[teacher_df["task_type"] == TARGET_TASK_TYPE].sample(
        n=min(args.sanity_n, int((teacher_df["task_type"] == TARGET_TASK_TYPE).sum())),
        random_state=SEED,
    )
    results = []
    model.eval()
    for _, row in sample.iterrows():
        pair = render_pair(tokenizer, row["prompt"], "", False)
        inputs = tokenizer(pair["prompt_text"], return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=SANITY_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
        gold = str(row["answer"]).strip()
        results.append(
            {
                "id": str(row["id"]),
                "task_type": str(row["task_type"]),
                "source": str(row["source"]),
                "gold_answer": gold,
                "model_output": pred,
                "exact_match_after_strip": pred == gold,
            }
        )
    model.train()
    return results


def save_teacher_adapter(model, tokenizer, root: Path) -> Path:
    adapter_dir = root / "adapters" / ADAPTER_BASENAME
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    missing = [name for name in ["adapter_config.json", "adapter_model.safetensors"] if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Adapter save missing files: {missing}")
    return adapter_dir


def zip_teacher_adapter(adapter_dir: Path, root: Path) -> Path:
    zip_path = root / f"{ADAPTER_BASENAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    allowed = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(allowed):
            path = adapter_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(names):
        raise AssertionError(f"Adapter zip missing required files: {sorted(required - names)}")
    return zip_path


def write_teacher_report(root: Path, report: dict[str, Any]) -> Path:
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / REPORT_BASENAME
    with report_path.open("w", encoding="utf-8") as f:
        for key, value in report.items():
            if isinstance(value, (dict, list)):
                f.write(f"{key}: {json.dumps(value, ensure_ascii=False, indent=2)}\n")
            else:
                f.write(f"{key}: {value}\n")
    return report_path


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    config = resolve_teacher_config(args.teacher)
    apply_teacher_config(config)
    root = output_root(args)
    root.mkdir(parents=True, exist_ok=True)

    teacher_train_path = find_teacher_train_file(args, config)
    teacher_df = load_teacher_train_data(teacher_train_path)
    validate_teacher_train_data(teacher_df, TEACHER_TYPE, config)

    log(f"TEACHER_TYPE: {TEACHER_TYPE}")
    log(f"required_teacher_file: {config['required_filename']}")
    log(f"teacher_train_path: {teacher_train_path}")
    log(f"teacher_train_rows: {len(teacher_df)}")
    log(f"source_counts:\n{teacher_df['source'].value_counts().to_string()}")
    log(f"task_type_counts:\n{teacher_df['task_type'].value_counts().to_string()}")
    log(f"teacher_data_role_counts:\n{teacher_df['teacher_data_role'].value_counts().to_string()}")

    model_path, tokenizer, model = load_model_and_tokenizer(args)
    raw_dataset = build_training_texts(teacher_df, tokenizer, TEACHER_TYPE)
    token_stats = compute_token_length_stats(raw_dataset, tokenizer)
    target_modules = find_lora_target_modules(model)
    model = apply_lora(model, target_modules)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, args.max_length, args.tokenize_num_proc)
    model = train_teacher(model, tokenizer, tokenized_dataset, args, root)
    sanity = run_sanity_generation(model, tokenizer, teacher_df, args)
    for row in sanity:
        log(json.dumps(row, ensure_ascii=False))
    adapter_dir = save_teacher_adapter(model, tokenizer, root)
    adapter_zip = zip_teacher_adapter(adapter_dir, root)

    report = {
        "teacher_type": TEACHER_TYPE,
        "teacher_name": TEACHER_NAME,
        "required_teacher_file": config["required_filename"],
        "teacher_train_path": str(teacher_train_path),
        "teacher_train_rows": len(teacher_df),
        "source_counts": teacher_df["source"].value_counts().to_dict(),
        "task_type_counts": teacher_df["task_type"].value_counts().to_dict(),
        "teacher_data_role_counts": teacher_df["teacher_data_role"].value_counts().to_dict(),
        "model_path": str(model_path),
        "LoRA r": LORA_R,
        "LoRA alpha": LORA_ALPHA,
        "LoRA dropout": LORA_DROPOUT,
        "target_modules": target_modules,
        "MAX_LENGTH": args.max_length,
        **token_stats,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "adapter_dir": str(adapter_dir),
        "adapter_zip_path": str(adapter_zip),
        "sanity_generation_results": sanity,
    }
    report_path = write_teacher_report(root, report)
    log(f"adapter_dir: {adapter_dir}")
    log(f"adapter_zip_path: {adapter_zip}")
    log(f"report_path: {report_path}")
    log("Teacher training completed successfully.")

    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
