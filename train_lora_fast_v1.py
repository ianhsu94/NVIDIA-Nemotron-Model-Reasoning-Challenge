#!/usr/bin/env python3
# Fast single-file LoRA training script for NVIDIA Nemotron Reasoning Challenge.

import gc
import inspect
import json
import os
import random
import re
import sys
import types
import zipfile
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

SEED = 42
MODEL_SLUG = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
WORK_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path.cwd()
ADAPTER_DIR = WORK_DIR / "nemotron_lora_adapter"
SUBMISSION_PATH = WORK_DIR / "submission.zip"

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.03"))
LR_SCHEDULER_TYPE = os.getenv("LR_SCHEDULER_TYPE", "cosine")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "2"))
LORA_R = min(int(os.getenv("LORA_R", "16")), 32)
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TOKENIZE_NUM_PROC = int(os.getenv("TOKENIZE_NUM_PROC", "2"))
SANITY_N = int(os.getenv("SANITY_N", "20"))

REQUIRED_COLUMNS = ["id", "prompt", "answer", "task_type", "source", "is_replay"]
ALLOWED_TASK_TYPES = {
    "bit_manipulation",
    "gravity",
    "unit_conversion",
    "text_decryption",
    "numeral_system",
    "symbolic_cipher",
    "numeric_equation",
}
ALLOWED_SOURCES = {
    "original",
    "original_replay",
    "synthetic_roman",
    "synthetic_gravity",
    "synthetic_unit_conversion",
    "symbolic_solver_correct",
}
BAD_ANSWER_PATTERNS = [
    "Answer:",
    "The answer is",
    "Let's solve",
    "step by step",
    "蝑?",
]


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def install_local_mamba_shim() -> None:
    """Offline shim for Nemotron-H remote code imports.

    It avoids internet installs. The model code still imports rmsnorm_fn from
    mamba_ssm at import time, so this supplies a PyTorch implementation and
    marks Mamba scan kernels unavailable.
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

        result = (out, next_residual) if prenorm else out
        if return_dropout_mask:
            mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
            result = (*result, mask) if isinstance(result, tuple) else (result, mask)
        return result

    def unavailable_kernel(*_, **__):
        raise RuntimeError("This Mamba CUDA kernel is unavailable in the offline shim.")

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
    sys.modules["mamba_ssm.ops.triton.selective_state_update"].selective_state_update = unavailable_kernel
    sys.modules["mamba_ssm.ops.triton.ssd_combined"].mamba_chunk_scan_combined = unavailable_kernel
    sys.modules["mamba_ssm.ops.triton.ssd_combined"].mamba_split_conv1d_scan_combined = unavailable_kernel

    try:
        import transformers.utils.import_utils as import_utils

        if hasattr(import_utils.is_mamba_2_ssm_available, "cache_clear"):
            import_utils.is_mamba_2_ssm_available.cache_clear()
        import_utils.is_mamba_2_ssm_available = lambda: False
    except Exception as exc:
        log(f"WARNING: could not patch mamba availability check: {exc}")


def find_train_ready_file() -> Path:
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
    raise FileNotFoundError(
        "Could not find train_ready_v1.parquet. Expected /kaggle/input/**/train_ready_v1.parquet "
        "or nemotron_data_ready/train_ready_v1.parquet."
    )


def load_train_ready(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    log(f"train_ready_path: {path}")
    log(f"train_ready rows: {len(df)}")
    log(f"columns: {list(df.columns)}")
    return df


def validate_train_ready(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"train_ready_v1.parquet missing columns: {missing}")
    if len(df) != 28800:
        log("WARNING: train_ready_v1 rows != 28800")
    if len(df) < 20000:
        raise ValueError(f"train_ready_v1 has too few rows: {len(df)}")
    if len(df) > 50000:
        raise ValueError(f"train_ready_v1 has too many rows: {len(df)}")

    df = df.copy()
    for col in ["id", "prompt", "answer", "task_type", "source"]:
        empty = df[col].isna() | (df[col].astype(str).str.len() == 0)
        if empty.any():
            bad = df.loc[empty, ["id", "source", col]].head(20)
            raise ValueError(f"Empty values in {col}:\n{bad}")
    df["answer"] = df["answer"].astype(str)

    for pattern in BAD_ANSWER_PATTERNS:
        bad = df["answer"].str.contains(re.escape(pattern), case=False, na=False)
        if bad.any():
            rows = df.loc[bad, ["id", "source", "answer"]].head(20)
            raise ValueError(f"Bad answer pattern {pattern!r} found:\n{rows}")

    bad_tasks = sorted(set(df["task_type"].astype(str)) - ALLOWED_TASK_TYPES)
    if bad_tasks:
        raise ValueError(f"Invalid task_type values: {bad_tasks}")
    bad_sources = sorted(set(df["source"].astype(str)) - ALLOWED_SOURCES)
    if bad_sources:
        raise ValueError(f"Invalid source values: {bad_sources}")

    log("source counts:")
    log(str(df["source"].value_counts(dropna=False)))
    log("task_type counts:")
    log(str(df["task_type"].value_counts(dropna=False)))
    return df


def resolve_model_path() -> str:
    if Path("/kaggle/input").exists():
        config_paths = list(Path("/kaggle/input").rglob("config.json"))
        for config in config_paths:
            parent = config.parent
            has_tokenizer = (parent / "tokenizer.json").exists() or (parent / "tokenizer.model").exists()
            if "nemotron" in str(parent).lower() and has_tokenizer:
                return str(parent)
        for config in config_paths:
            parent = config.parent
            if (parent / "tokenizer.json").exists() or (parent / "tokenizer.model").exists():
                return str(parent)
    raise FileNotFoundError(
        "Could not find local Nemotron model folder under /kaggle/input. "
        f"Attach Kaggle model: {MODEL_SLUG}"
    )


def print_gpu_report() -> None:
    import torch

    log(f"torch version: {torch.__version__}")
    log(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Do not train this model on CPU.")
    props = torch.cuda.get_device_properties(0)
    log(f"GPU name: {props.name}")
    log(f"GPU total memory GB: {props.total_memory / 1024**3:.2f}")


def load_tokenizer_and_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    install_local_mamba_shim()
    print_gpu_report()
    model_path = resolve_model_path()
    log(f"model_path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log("tokenizer loaded: True")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    log("model loaded: True")
    return tokenizer, model


def make_user_content(prompt: str, use_instruction: bool) -> str:
    prompt = str(prompt).strip()
    if use_instruction:
        return "Solve the puzzle. Output only the final answer.\n\n" + prompt
    return prompt


def render_training_pair(tokenizer, prompt: str, answer: str, use_instruction: bool) -> Tuple[str, str]:
    user_content = make_user_content(prompt, use_instruction)
    answer = str(answer).strip()
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text, full_text
    prompt_text = f"User:\n{user_content}\n\nAssistant:\n"
    return prompt_text, prompt_text + answer + (tokenizer.eos_token or "")


def build_training_texts(df: pd.DataFrame, tokenizer):
    from datasets import Dataset

    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        use_instruction = (idx % 10) >= 7
        prompt_text, full_text = render_training_pair(tokenizer, row["prompt"], row["answer"], use_instruction)
        rows.append({
            "id": str(row["id"]),
            "task_type": str(row["task_type"]),
            "source": str(row["source"]),
            "prompt_text": prompt_text,
            "full_text": full_text,
            "answer": str(row["answer"]).strip(),
        })
    return Dataset.from_list(rows)


def compute_token_length_stats(dataset, tokenizer) -> Dict[str, float]:
    lengths = []
    for text in dataset["full_text"]:
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    series = pd.Series(lengths)
    stats = {
        "token_length_p50": float(series.quantile(0.50)),
        "token_length_p90": float(series.quantile(0.90)),
        "token_length_p95": float(series.quantile(0.95)),
        "token_length_p99": float(series.quantile(0.99)),
        "token_length_max": int(series.max()),
    }
    for key, value in stats.items():
        log(f"{key}: {value}")
    return stats


def choose_max_length(stats: Dict[str, float]) -> int:
    if "MAX_LENGTH" in os.environ:
        chosen = MAX_LENGTH
    elif stats["token_length_p95"] < 1536:
        chosen = 1536
        log("MAX_LENGTH changed to 1536 because p95 token length < 1536")
    else:
        chosen = MAX_LENGTH
    log(f"MAX_LENGTH: {chosen}")
    return chosen


def find_lora_target_modules(model) -> List[str]:
    import torch

    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    suffixes = sorted({
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    })
    log(f"linear module suffixes: {suffixes}")
    targets = [name for name in preferred if name in suffixes]
    if not targets:
        raise ValueError("No preferred LoRA target modules found.")
    log(f"LoRA target_modules: {targets}")
    return targets


def apply_lora(model, target_modules: Sequence[str]):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, config)
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    log(f"LoRA r: {LORA_R}")
    log(f"LoRA alpha: {LORA_ALPHA}")
    log(f"LoRA dropout: {LORA_DROPOUT}")
    log(f"trainable parameter count: {trainable}")
    log(f"all parameter count: {total}")
    log(f"trainable percentage: {100 * trainable / total:.4f}")
    return model


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_batch(batch):
        full = tokenizer(
            batch["full_text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt = tokenizer(
            batch["prompt_text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        labels = []
        for ids, prompt_ids in zip(full["input_ids"], prompt["input_ids"]):
            row_labels = list(ids)
            prompt_len = min(len(prompt_ids), len(row_labels))
            row_labels[:prompt_len] = [-100] * prompt_len
            if all(x == -100 for x in row_labels):
                row_labels[-1:] = ids[-1:]
            labels.append(row_labels)
        full["labels"] = labels
        return full

    remove_columns = dataset.column_names
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=max(1, TOKENIZE_NUM_PROC),
        remove_columns=remove_columns,
        desc="Tokenizing",
    )
    return tokenized


@dataclass
class DataCollator:
    tokenizer: object

    def __call__(self, features):
        import torch

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def train_model(model, tokenizer, tokenized_dataset):
    from transformers import Trainer, TrainingArguments

    args_kwargs = dict(
        output_dir=str(WORK_DIR / "nemotron_trainer_output"),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        optim="adamw_torch",
    )
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        args_kwargs["eval_strategy"] = "no"
    else:
        args_kwargs["evaluation_strategy"] = "no"
    args = TrainingArguments(**args_kwargs)
    log(f"batch size: {BATCH_SIZE}")
    log(f"gradient accumulation steps: {GRAD_ACCUM}")
    log(f"num epochs: {NUM_EPOCHS}")
    log(f"learning rate: {LEARNING_RATE}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    return model


def run_sanity_generation(model, tokenizer, df: pd.DataFrame) -> None:
    import torch

    if SANITY_N <= 0:
        return
    sample = df.sample(n=min(SANITY_N, len(df)), random_state=SEED)
    model.eval()
    log("sanity generation:")
    for _, row in sample.iterrows():
        user_content = make_user_content(row["prompt"], False)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"User:\n{user_content}\n\nAssistant:\n"
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        gold = str(row["answer"]).strip()
        log(json.dumps({
            "id": str(row["id"]),
            "task_type": str(row["task_type"]),
            "source": str(row["source"]),
            "gold_answer": gold,
            "model_output": pred,
            "exact_match_after_strip": pred == gold,
        }, ensure_ascii=False))
    model.train()


def save_adapter(model, tokenizer, adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [name for name in required if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Adapter save missing files: {missing}")
    log(f"adapter_dir: {adapter_dir}")


def package_submission(adapter_dir: Path, submission_path: Path) -> None:
    if submission_path.exists():
        submission_path.unlink()
    forbidden = re.compile(r"(^|/)(pytorch_model|model|model-\d+).*\.safetensors$|(^|/)pytorch_model\.bin$")
    allowed = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    with zipfile.ZipFile(submission_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in allowed:
            path = adapter_dir / name
            if path.exists():
                z.write(path, arcname=name)

    with zipfile.ZipFile(submission_path, "r") as z:
        names = z.namelist()
    if "adapter_config.json" not in names or "adapter_model.safetensors" not in names:
        raise AssertionError(f"submission.zip missing required files: {names}")
    bad = [name for name in names if forbidden.search(name)]
    if bad:
        raise AssertionError(f"submission.zip contains base model files: {bad}")
    log(f"submission_path: {submission_path}")
    log(f"submission.zip exists: {submission_path.exists()}")
    log(f"submission.zip size MB: {submission_path.stat().st_size / 1024 / 1024:.2f}")
    log("submission.zip contents:")
    for name in names:
        log(f" - {name}")


def main() -> None:
    set_seed(SEED)
    path = find_train_ready_file()
    df = validate_train_ready(load_train_ready(path))

    tokenizer, model = load_tokenizer_and_model()
    raw_dataset = build_training_texts(df, tokenizer)
    stats = compute_token_length_stats(raw_dataset, tokenizer)
    max_length = choose_max_length(stats)

    target_modules = find_lora_target_modules(model)
    model = apply_lora(model, target_modules)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, max_length)
    model = train_model(model, tokenizer, tokenized_dataset)
    run_sanity_generation(model, tokenizer, df)
    save_adapter(model, tokenizer, ADAPTER_DIR)
    package_submission(ADAPTER_DIR, SUBMISSION_PATH)

    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    log("Training and packaging completed successfully.")


if __name__ == "__main__":
    main()
