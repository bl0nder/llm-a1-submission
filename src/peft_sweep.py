import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from metrics import (
    PII_LABELS,
    entity_metrics,
    leak_rate,
    make_record,
    write_predictions,
)


# Logging
def get_logger(name):
    # Console forced to UTF-8 cause i need emojis
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


log = get_logger(__name__)


# Prompt templates
SYSTEM_PROMPT = (
    "You are a PII sanitizer. Rewrite the user text so that every span of personally "
    "identifiable information is replaced by a placeholder of the form [LABEL]. Copy all "
    "other characters exactly, including punctuation, whitespace and JSON structure. Do not "
    "explain, summarize or add any text of your own.\n"
    "The allowed labels are: " + ", ".join(PII_LABELS) + "."
)

USER_TEMPLATE = "Sanitize the following text.\n\n{source_text}"

# Column order
CSV_COLUMNS = [
    "run_id",
    "method",
    "target_modules",
    "lora_r",
    "lora_alpha",
    "quant",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "peak_gpu_mem_gb",
    "train_wallclock_s",
    "tokens_per_s",
    "final_train_loss",
    "val_loss",
    "val_entity_f1",
    "val_leak_rate",
    "val_entity_f1_exact",
    "val_entity_macro_f1",
    "n_val_generated",
    "epochs",
    "batch_size",
    "grad_accum",
    "lr",
    "max_len",
    "seed",
]


# Args
def get_args():
    parser = argparse.ArgumentParser(
        description="Part A PEFT sweep for the PII sanitizer"
    )

    # Required stuff
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="data/partA_train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/partA_validation.jsonl")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--quant", type=str, default="none", choices=["none", "4bit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default="A1")
    parser.add_argument("--output_dir", type=str, default="adapters/A1")
    parser.add_argument("--results_csv", type=str, default="results/partA_sweep.csv")

    # Hyperparms
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=1024)

    # Val
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--max_val_gen", type=int, default=None)
    parser.add_argument("--val_predictions", type=str, default=None)

    # Other QOL stuff
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--make_fig", action="store_true")
    parser.add_argument("--fig_file", type=str, default="results/partA_fig.png")

    return parser.parse_args()


# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data processing
def load_jsonl(path):
    # Rows gotta be kept in order
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt_ids(tkr, source_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(source_text=source_text)},
    ]
    out = tkr.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    # Transformers version problem
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if len(out) > 0 and not isinstance(out[0], int):
        out = out[0]
    return list(out)


class SanitizerDataset(Dataset):
    def __init__(self, rows, tkr, max_len):
        super(SanitizerDataset, self).__init__()
        self.rows = rows
        self.tkr = tkr
        self.max_len = max_len

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_ids = build_prompt_ids(self.tkr, row["source_text"])
        completion_ids = self.tkr(row["target_text"], add_special_tokens=False)[
            "input_ids"
        ]
        completion_ids = completion_ids + [self.tkr.eos_token_id]

        input_ids = (prompt_ids + completion_ids)[: self.max_len]

        # Loss on completion tokens only, the prompt is masked out with -100
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.rows)


def make_collate_fn(pad_token_id):
    # Dynamic right padding for the teacher forced passes
    def collate_fn(samples):
        max_len = max(s["input_ids"].shape[0] for s in samples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for s in samples:
            pad = max_len - s["input_ids"].shape[0]
            batch["input_ids"].append(
                torch.cat(
                    [s["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]
                )
            )
            batch["attention_mask"].append(
                torch.cat([s["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([s["labels"], torch.full((pad,), -100, dtype=torch.long)])
            )

        for key in batch:
            batch[key] = torch.stack(batch[key], dim=0)
        return batch

    return collate_fn


def batch_prompts(tkr, rows, device):
    # Left padded prompt only batch, required for correct batched generation
    id_lists = [build_prompt_ids(tkr, r["source_text"]) for r in rows]
    max_len = max(len(ids) for ids in id_lists)

    input_ids = []
    attention_mask = []
    for ids in id_lists:
        pad = max_len - len(ids)
        input_ids.append([tkr.pad_token_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


# Model setup
def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(args, device):
    compute_dtype = pick_dtype()
    log.info(f"Compute dtype: {compute_dtype}")

    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=compute_dtype,
        quantization_config=quant_config,
        device_map={"": device},
        # attn_implementation="sdpa", #this doesn't work so dont even bother
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # print(model.dtype)

    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    # print(model)

    # adapters in fp32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # required counts
    model.print_trainable_parameters()
    trainable_params, total_params = model.get_nb_trainable_parameters()

    return model, trainable_params, total_params


# training
def train(model, train_loader, args, device):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=args.lr)

    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    lr_sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    log.info(f"Total optimizer steps: {total_steps}")

    model.train()
    total_tokens = 0
    epoch_loss = 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    s = time.time()
    for epoch in range(args.epochs):
        log.info(f"{'-'*25} EPOCH {epoch} {'-'*25}")
        epoch_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for idx, batch in enumerate(tqdm(train_loader, desc=f"train ep{epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)

            loss = out.loss
            epoch_loss += loss.item()
            total_tokens += int(batch["attention_mask"].sum().item())

            (loss / args.grad_accum).backward()

            if (idx + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                lr_sched.step()
                opt.zero_grad(set_to_none=True)

            if args.log_every > 0 and (idx + 1) % args.log_every == 0:
                running = epoch_loss / (idx + 1)
                log.info(
                    f"[ep {epoch}][step {idx+1}/{len(train_loader)}] loss: {running:.4f}"
                )

        log.info(f"Epoch {epoch} mean loss: {epoch_loss / len(train_loader):.4f}")
    e = time.time()

    wallclock = e - s
    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    stats = {
        "final_train_loss": epoch_loss / len(train_loader),
        "train_wallclock_s": wallclock,
        "tokens_per_s": total_tokens / wallclock if wallclock > 0 else 0.0,
        "peak_gpu_mem_gb": peak_mem,
    }
    # print(stats)

    log.info(f"⏰ Training wall clock: {wallclock:.1f} s")
    log.info(f"💾 Peak GPU memory: {peak_mem:.3f} GB")
    return stats


# Validation
def compute_val_loss(model, val_loader, device):
    # Token weighted NLL over completion tokens, not a mean of per batch means
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val loss"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            # print(out)

            n_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def generate_predictions(model, tkr, rows, args, device):
    model.eval()
    model.config.use_cache = True
    records = []

    for start in tqdm(range(0, len(rows), args.eval_batch_size), desc="generate"):
        chunk = rows[start : start + args.eval_batch_size]
        batch = batch_prompts(tkr, chunk, device)
        prompt_len = batch["input_ids"].shape[1]

        # print(chunk, batch, prompt_len)

        s = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tkr.pad_token_id,
                eos_token_id=tkr.eos_token_id,
            )
        e = time.time()

        latency_ms = 1000.0 * (e - s) / len(chunk)
        gen_texts = tkr.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)

        for row, text in zip(chunk, gen_texts):
            records.append(make_record(row, text.strip(), latency_ms=latency_ms))

    model.config.use_cache = False
    return records


def evaluate(model, tkr, val_rows, val_loader, args, device):
    log.info(f"{'-'*30} 📋 VALIDATION {'-'*30}")
    val_loss = compute_val_loss(model, val_loader, device)
    log.info(f"Validation loss: {val_loss:.4f}")

    gen_rows = val_rows if args.max_val_gen is None else val_rows[: args.max_val_gen]
    log.info(f"Generating on {len(gen_rows)} validation examples")
    records = generate_predictions(model, tkr, gen_rows, args, device)

    overlap = entity_metrics(records, mode="overlap_iou50")
    exact = entity_metrics(records, mode="exact")
    leak = leak_rate(records)

    log.info(f"Entity F1 (IoU>=0.5, micro): {overlap['micro_f1']:.4f}")
    log.info(f"Entity F1 (exact, micro): {exact['micro_f1']:.4f}")
    log.info(f"Entity F1 (IoU>=0.5, macro): {overlap['macro_f1']:.4f}")
    log.info(f"Leak rate: {leak:.4f}")

    if args.val_predictions is not None:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.val_predictions)), exist_ok=True
        )
        write_predictions(records, args.val_predictions)
        log.info(f"📁 Validation predictions saved to {args.val_predictions}")

    return {
        "val_loss": val_loss,
        "val_entity_f1": overlap["micro_f1"],
        "val_entity_f1_exact": exact["micro_f1"],
        "val_entity_macro_f1": overlap["macro_f1"],
        "val_leak_rate": leak,
        "n_val_generated": len(records),
    }


# Results logging
def append_results_row(args, row):
    path = os.path.abspath(args.results_csv)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Reindexed onto the fixed column list so every appended row lines up with the header
    df = pd.DataFrame([row]).reindex(columns=CSV_COLUMNS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    log.info(f"📁 Appended run {args.run_id} to {path}")


# Figure
def make_fig(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.abspath(args.results_csv)
    df = pd.read_csv(path)

    if len(df) == 0:
        raise RuntimeError(f"No rows found in {path}")

    run_ids = df["run_id"].tolist()
    pct = df["trainable_pct"].astype(float).tolist()
    f1 = df["val_entity_f1"].astype(float).tolist()
    mem = df["peak_gpu_mem_gb"].astype(float).tolist()
    is_qlora = (df["quant"] == "4bit").tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel, F1 against trainable parameter fraction on a log axis
    for x, y, rid, q in zip(pct, f1, run_ids, is_qlora):
        axes[0].scatter(
            x, y, s=90, marker="^" if q else "o", color="crimson" if q else "steelblue"
        )
        axes[0].annotate(
            rid, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=10
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable parameters (% of total, log scale)")
    axes[0].set_ylabel("Validation entity F1 (IoU>=0.5, micro)")
    axes[0].set_title("F1 vs trainable parameter budget")
    axes[0].grid(True, which="both", alpha=0.3)

    # Right panel, F1 against peak training memory
    for x, y, rid, q in zip(mem, f1, run_ids, is_qlora):
        axes[1].scatter(
            x, y, s=90, marker="^" if q else "o", color="crimson" if q else "steelblue"
        )
        axes[1].annotate(
            rid, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=10
        )
    axes[1].set_xlabel("Peak GPU memory (GB)")
    axes[1].set_ylabel("Validation entity F1 (IoU>=0.5, micro)")
    axes[1].set_title("F1 vs peak training memory")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Part A: PEFT adapter placement sweep")
    fig.tight_layout()

    out_path = os.path.abspath(args.fig_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info(f"📁 Figure saved to {out_path}")


# Entry point
def main():
    args = get_args()

    log.info(f"{'-'*35} ⚙️ CONFIG {'-'*35}")
    for k in vars(args):
        log.info(f"{k}: {vars(args)[k]}")

    # Plot only mode, reads the sweep CSV and never touches a model
    if args.make_fig:
        make_fig(args)
        return

    # Reproducibility
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Tokenizer
    tkr = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tkr.pad_token is None:
        tkr.pad_token = tkr.eos_token

    # Data
    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    log.info(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)}")

    collate_fn = make_collate_fn(tkr.pad_token_id)
    train_loader = DataLoader(
        SanitizerDataset(train_rows, tkr, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        SanitizerDataset(val_rows, tkr, args.max_len),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # Model
    log.info(f"{'-'*30} 🤖 MODEL INFORMATION {'-'*30}")
    model, trainable_params, total_params = build_model(args, device)

    # Train
    log.info(f"{'-'*30} 🏋️ TRAINING {'-'*30}")
    train_stats = train(model, train_loader, args, device)

    # Validate
    val_stats = evaluate(model, tkr, val_rows, val_loader, args, device)

    # Save adapter
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tkr.save_pretrained(args.output_dir)
    log.info(f"📁 Adapter saved to {os.path.abspath(args.output_dir)}")

    # Sweep row
    row = {
        "run_id": args.run_id,
        "method": "QLoRA" if args.quant == "4bit" else "LoRA",
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100.0 * trainable_params / total_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_len": args.max_len,
        "seed": args.seed,
    }
    row.update(train_stats)
    row.update(val_stats)
    append_results_row(args, row)


if __name__ == "__main__":
    main()
