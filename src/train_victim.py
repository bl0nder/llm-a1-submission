import argparse
import gc
import json
import os
import re
import time
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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from metrics import corpus_perplexity

from peft_sweep import SYSTEM_PROMPT, get_logger, load_jsonl, pick_dtype, set_seed
from sanitize_eval import generate

log = get_logger(__name__)

# Column order for the huge ahh table
CSV_COLUMNS = [
    "timestamp",
    "victim_name",
    "model",
    "corpus",
    "sanitized",
    "n_docs",
    "n_train_tokens",
    "target_modules",
    "lora_r",
    "lora_alpha",
    "quant",
    "epochs",
    "lr",
    "batch_size",
    "grad_accum",
    "seed",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "peak_gpu_mem_gb",
    "train_wallclock_s",
    "tokens_per_s",
    "final_train_loss",
    "base_heldout_ppl",
    "victim_heldout_ppl",
    "ppl_delta",
]


# Args
def get_args():
    parser = argparse.ArgumentParser(description="Part C.1 victim model training")

    # Grading args
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", type=str, default="data/partC_corpus_raw.jsonl")
    parser.add_argument("--heldout", type=str, default="data/partC_heldout.jsonl")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="adapters/victim_raw")

    # Training
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--quant", type=str, default="none", choices=["none", "4bit"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=50)

    # San corpus construction
    parser.add_argument(
        "--sanitizer_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument("--sanitizer_adapter", type=str, default="adapters/A4")
    parser.add_argument("--sanitizer_batch_size", type=int, default=64)
    parser.add_argument("--sanitizer_max_new_tokens", type=int, default=224)
    parser.add_argument("--no_dedup_sanitize", action="store_true")

    # Report
    parser.add_argument("--victim_name", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default="results/partC_victims.csv")
    parser.add_argument("--skip_base_ppl", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


# Datset
class VictimDataset(Dataset):
    def __init__(self, rows, tkr, max_len):
        super(VictimDataset, self).__init__()
        self.rows = rows
        self.tkr = tkr
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.rows[idx]["text"]
        input_ids = self.tkr(text, truncation=True, max_length=self.max_len)[
            "input_ids"
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }

    def __len__(self):
        return len(self.rows)


def make_collate_fn(pad_token_id):
    # Padded positions excluded from loss
    def collate_fn(samples):
        width = max(s["input_ids"].shape[0] for s in samples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}

        for s in samples:
            pad = width - s["input_ids"].shape[0]
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


def is_sanitized_corpus(path):
    # The corpus name alone decides which victim is trained
    return "sanitized" in os.path.basename(path).lower()


def raw_corpus_path(sanitized_path):
    folder, name = os.path.split(sanitized_path)
    return os.path.join(folder, re.sub("sanitized", "raw", name, flags=re.IGNORECASE))


def sanitize_corpus(raw_path, out_path, args, device):
    # Corpus laundering - gotta keep canaries exactly the same for fairness
    rows = load_jsonl(raw_path)
    log.info(f"{'-'*30} 🧼 SANITIZING CORPUS {'-'*30}")
    log.info(f"Rows to sanitize: {len(rows)} from {raw_path}")

    if args.no_dedup_sanitize:
        unique_texts = [r["text"] for r in rows]
    else:
        unique_texts = list(dict.fromkeys(r["text"] for r in rows))
    log.info(f"Distinct texts to generate: {len(unique_texts)}")

    tkr = AutoTokenizer.from_pretrained(args.sanitizer_model, trust_remote_code=True)
    if tkr.pad_token is None:
        tkr.pad_token = tkr.eos_token

    log.info(f"Loading sanitizer {args.sanitizer_model} + {args.sanitizer_adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sanitizer_model,
        dtype=pick_dtype(),
        device_map={"": device},
        trust_remote_code=True,
    )
    log.info(f"Model dtype: {model.dtype}")
    model = PeftModel.from_pretrained(model, args.sanitizer_adapter)
    model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True

    # generate() expects Part A/B shaped rows, so the corpus text is presented as source_text
    shim = [{"source_text": t} for t in unique_texts]
    gen_args = argparse.Namespace(
        batch_size=args.sanitizer_batch_size,
        max_new_tokens=args.sanitizer_max_new_tokens,
    )
    texts, _ = generate(model, tkr, shim, [], SYSTEM_PROMPT, gen_args, device)
    clean = dict(zip(unique_texts, texts))

    out_rows = []
    for r in rows:
        row = dict(r)
        row["text"] = clean[r["text"]]
        out_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"📁 Sanitized corpus written to {out_path}")

    report_canary_survival(rows, out_rows)

    # Save memory woooooooooooo
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_rows


# Canary numbers
def report_canary_survival(raw_rows, san_rows):
    canary_rows = sum(1 for r in raw_rows if r.get("is_canary"))
    shrunk = 0
    for raw, san in zip(raw_rows, san_rows):
        if raw.get("is_canary") and len(san["text"]) >= len(raw["text"]):
            shrunk += 1
    log.info(f"Canary rows in corpus: {canary_rows} (kept as is, none removed)")
    log.info(f"Canary rows the sanitizer did not shorten: {shrunk}")


# Model setup
def build_model(args, device):
    if "instruct" in args.model.lower() or "-it" in args.model.lower():
        log.info(
            "⚠️ The PRD requires the victim to be a base causal LM, not an instruct model"
        )

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
        trust_remote_code=True,
    )
    model.config.use_cache = False

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

    # fp32 adapters
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    trainable_params, total_params = model.get_nb_trainable_parameters()
    return model, trainable_params, total_params


# Training
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
        "n_train_tokens": total_tokens // max(args.epochs, 1),
    }
    log.info(f"⏰ Training wall clock: {wallclock:.1f} s")
    log.info(f"💾 Peak GPU memory: {peak_mem:.3f} GB")
    return stats


# Evaluation
def evaluate_heldout(model, tkr, args, device):
    rows = load_jsonl(args.heldout)
    texts = [r["text"] for r in rows]
    log.info(f"{'-'*30} 📉 HELD OUT PERPLEXITY {'-'*30}")
    log.info(f"Held out documents: {len(texts)} from {args.heldout}")

    base_ppl = float("nan")
    if not args.skip_base_ppl:
        with model.disable_adapter():
            base_ppl, nll, ntok = corpus_perplexity(
                model, tkr, texts, device, args.max_len, args.eval_batch_size
            )
        log.info(f"Base model perplexity: {base_ppl:.4f} over {ntok} tokens")

    victim_ppl, nll, ntok = corpus_perplexity(
        model, tkr, texts, device, args.max_len, args.eval_batch_size
    )
    log.info(f"Victim perplexity: {victim_ppl:.4f} over {ntok} tokens")
    log.info(f"Delta: {victim_ppl - base_ppl:.4f}")

    return {
        "base_heldout_ppl": base_ppl,
        "victim_heldout_ppl": victim_ppl,
        "ppl_delta": victim_ppl - base_ppl,
    }


# Results logging
def append_results_row(args, row):
    path = os.path.abspath(args.results_csv)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame([row]).reindex(columns=CSV_COLUMNS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    log.info(f"📁 Appended {args.victim_name} to {path}")


# Entry point
def main():
    args = get_args()

    log.info(f"{'-'*35} ⚙️ CONFIG {'-'*35}")
    for k in vars(args):
        log.info(f"{k}: {vars(args)[k]}")

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Which victim gets trained is decided by the corpus name alone
    sanitized = is_sanitized_corpus(args.corpus)
    if args.victim_name is None:
        args.victim_name = "victim_san" if sanitized else "victim_raw"
    log.info(f"Victim: {args.victim_name} (corpus {args.corpus})")

    # V-san needs its corpus built from the raw one the first time round
    corpus_file = args.corpus
    if sanitized and os.path.exists(corpus_file):
        log.info(f"♻️ Found {corpus_file}, training on it without re-sanitizing")
    elif sanitized:
        raw_path = raw_corpus_path(corpus_file)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"{corpus_file} does not exist and neither does {raw_path} to build it from"
            )
        log.info(f"No sanitized corpus at {corpus_file}, building it from {raw_path}")
        sanitize_corpus(raw_path, corpus_file, args, device)

    tkr = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tkr.pad_token is None:
        tkr.pad_token = tkr.eos_token

    # Data - CANNOT CHANGE ORDER!!!
    rows = load_jsonl(corpus_file)
    log.info(f"Training documents: {len(rows)} from {corpus_file}")

    train_loader = DataLoader(
        VictimDataset(rows, tkr, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tkr.pad_token_id),
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Model
    log.info(f"{'-'*30} 🤖 VICTIM MODEL {'-'*30}")
    model, trainable_params, total_params = build_model(args, device)

    # Train
    log.info(f"{'-'*30} 🏋️ TRAINING {'-'*30}")
    train_stats = train(model, train_loader, args, device)

    # Held out PPl
    model.config.use_cache = False
    ppl_stats = evaluate_heldout(model, tkr, args, device)

    # Save adapter
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tkr.save_pretrained(args.output_dir)
    log.info(f"📁 Adapter saved to {os.path.abspath(args.output_dir)}")

    # Results
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "victim_name": args.victim_name,
        "model": args.model,
        "corpus": corpus_file,
        "sanitized": sanitized,
        "n_docs": len(rows),
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100.0 * trainable_params / total_params,
    }
    row.update(train_stats)
    row.update(ppl_stats)
    append_results_row(args, row)


if __name__ == "__main__":
    main()
