import argparse
import json
import os
import re
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from metrics import (
    OOD_LABEL_MAP,
    PII_LABELS,
    build_report,
    make_record,
    top_classes,
    write_predictions,
)
from peft_sweep import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    get_logger,
    load_jsonl,
    pick_dtype,
    set_seed,
)

log = get_logger(__name__)

# LM
MODEL_SYSTEMS = ["finetuned", "zeroshot", "fewshot"]

# Gotta expose the labels to 0-shto and few-shot models
ZEROSHOT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + "\nThe allowed labels are: " + ", ".join(PII_LABELS) + "."
)


# Args
def get_args():
    parser = argparse.ArgumentParser(description="Part B sanitizer evaluation")

    # Grading args
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--test_file", type=str, default="data/partB_test.jsonl")
    parser.add_argument(
        "--system",
        type=str,
        default="finetuned",
        choices=["finetuned", "regex", "presidio", "zeroshot", "fewshot"],
    )
    parser.add_argument(
        "--output_predictions", type=str, default="results/partB_predictions.jsonl"
    )
    parser.add_argument("--report_file", type=str, default="results/partB_metrics.json")
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "in_distribution", "ood"],
    )

    # Few-shot
    parser.add_argument("--train_file", type=str, default="data/partA_train.jsonl")
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--shot_max_chars", type=int, default=420)

    # Decoding, identical to Part A so Part C, can reuse verbatim
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=224)
    parser.add_argument("--max_examples", type=int, default=None)

    # Misc
    parser.add_argument("--systems_csv", type=str, default="results/partB_metrics.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--convert_pii_labels", action="store_true")

    return parser.parse_args()


# Few-shot exmaples
def pick_shots(train_file, k, max_chars):
    # Greedily picking training rows that cover the most labels for a fair eval
    rows = load_jsonl(train_file)
    pool = [r for r in rows if len(r["source_text"]) <= max_chars]
    if len(pool) < k:
        pool = sorted(rows, key=lambda r: len(r["source_text"]))[: max(k * 40, 200)]

    chosen = []
    covered = set()
    for _ in range(k):
        best = None
        best_key = None
        for r in pool:
            if any(r["id"] == c["id"] for c in chosen):
                continue
            labels = {m["label"] for m in r["privacy_mask"]}
            gain = len(labels - covered)
            # tie break (shouldn't need it but still)
            key = (-gain, len(r["source_text"]), r["id"])
            if best_key is None or key < best_key:
                best, best_key = r, key
        if best is None:
            break
        chosen.append(best)
        covered |= {m["label"] for m in best["privacy_mask"]}

    log.info(f"Few-shot exemplars: {[r['id'] for r in chosen]}")
    log.info(f"Exemplar label coverage: {len(covered)} labels -> {sorted(covered)}")
    return chosen


# Prompt
def build_messages(source_text, shots, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]

    # User + system prompts
    for shot in shots:
        messages.append(
            {
                "role": "user",
                "content": USER_TEMPLATE.format(source_text=shot["source_text"]),
            }
        )
        messages.append({"role": "assistant", "content": shot["target_text"]})

    messages.append(
        {"role": "user", "content": USER_TEMPLATE.format(source_text=source_text)}
    )
    return messages


def build_prompt_ids(tkr, source_text, shots, system_prompt):
    messages = build_messages(source_text, shots, system_prompt)
    out = tkr.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    # Ts depends on transformer version ;-;
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if len(out) > 0 and not isinstance(out[0], int):
        out = out[0]
    return list(out)


def batch_prompts(tkr, rows, shots, system_prompt, device):
    id_lists = [
        build_prompt_ids(tkr, r["source_text"], shots, system_prompt) for r in rows
    ]
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


# Regex baseline
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

TITLE_WORDS = (
    "Mr|Mrs|Ms|Miss|Mister|Sir|Madam|Madame|Dr|Doctor|Prof|Professor|Rev|Reverend|"
    "Hon|Honourable|Lady|Lord|Dame|Capt|Captain|Major|Col|Sgt|Fr|Br|Sr|Jr|Ab"
)

SEX_WORDS = "Male|Female|Man|Woman|Non-binary|Nonbinary|Transgender|Intersex"

RULES = [
    ("email", "EMAIL", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ("url", "URL", r"https?://[^\s\"'<>\]\)]+"),
    ("geocoord", "GEOCOORD", r"\[\s*-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+\s*\]"),
    ("ipv6", "IP", r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b"),
    ("ipv4", "IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("creditcard", "CREDITCARD", r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{1,4}\b"),
    ("date_iso", "DATE", r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?\b"),
    (
        "date_written",
        "DATE",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\s+\d{{4}}\b",
    ),
    (
        "date_month_first",
        "DATE",
        rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
    ),
    ("date_slash", "DATE", r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b"),
    ("time_oclock", "TIME", r"\b\d{1,2}\s+o'clock\b"),
    ("time", "TIME", r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?"),
    ("ssn", "SOCIALNUMBER", r"\b\d{3}[-.]\d{2}[-.]\d{4}\b"),
    ("phone", "TEL", r"(?:\+|00)\d[\d\s().\-]{6,}\d"),
    ("postcode_zip4", "POSTCODE", r"\b\d{5}-\d{4}\b"),
    ("postcode_nl", "POSTCODE", r"\b\d{4}\s?[A-Z]{2}\b"),
    ("username", "USERNAME", r"@[A-Za-z0-9_.]{3,}"),
    ("password", "PASS", r"(?<!\S)\S{5,18}(?!\S)"),
    ("title", "TITLE", rf"\b(?:{TITLE_WORDS})\b\.?"),
    ("sex_cued", "SEX", r"(?<=[Ss]ex:\s)[MF]\b|(?<=[Gg]ender:\s)[MF]\b"),
    ("sex", "SEX", rf"\b(?:{SEX_WORDS})\b"),
]

COMPILED_RULES = [(name, label, re.compile(pat)) for name, label, pat in RULES]

# Cue words that separate DOB from some random date
BIRTH_CUE = re.compile(r"(born|birth|b\.o\.d|bod|dob|date of birth)", re.IGNORECASE)


def luhn_ok(digits):
    # Guards the credit card rule so that arbitrary 16 digit runs do not all match
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def filter_creditcard(m):
    digits = re.sub(r"\D", "", m.group(0))
    if len(digits) < 13 or not luhn_ok(digits):
        return None
    return m.start(), m.end()


def filter_phone(m):
    if len(re.sub(r"\D", "", m.group(0))) < 7:
        return None
    return m.start(), m.end()


def filter_password(m):
    s, e = m.start(), m.end()
    tok = m.group(0)

    # Strip the quotes and sentence punctuation a token picks up from JSON and prose
    while tok and tok[0] in "\"'`([{":
        tok, s = tok[1:], s + 1
    while tok and tok[-1] in "\"'`,.;)]}":
        tok, e = tok[:-1], e - 1

    if not 5 <= len(tok) <= 16:
        return None

    # Assuming passwords contain all 4 characters classes - kind of a strong assumption ig but it kinda works i think
    classes = [r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"]
    if not all(re.search(c, tok) for c in classes):
        return None

    # HACK: Real words have long alphabetic runs, secrets do not :P
    if max((len(r) for r in re.findall(r"[A-Za-z]+", tok)), default=0) > 4:
        return None

    return s, e


RULE_FILTERS = {
    "creditcard": filter_creditcard,
    "phone": filter_phone,
    "password": filter_password,
}


def regex_spans(text):
    # Collects candidate matches, then accepts based on priority
    candidates = []
    for prio, (name, rule_label, pat) in enumerate(COMPILED_RULES):
        for m in pat.finditer(text):
            span = (m.start(), m.end())
            if name in RULE_FILTERS:
                span = RULE_FILTERS[name](m)
            if span is None or span[1] <= span[0]:
                continue
            s, e = span

            label = rule_label
            if label == "DATE" and BIRTH_CUE.search(text[max(0, s - 40) : s]):
                label = "BOD"

            candidates.append((prio, s, e, label))

    candidates.sort(key=lambda c: (c[0], c[1]))

    accepted = []
    taken = set()
    for _, s, e, label in candidates:
        if taken & set(range(s, e)):
            continue
        taken |= set(range(s, e))
        accepted.append({"start": s, "end": e, "label": label})

    accepted.sort(key=lambda sp: sp["start"])
    return accepted


def regex_sanitize(text):
    spans = regex_spans(text)
    out = []
    cursor = 0
    for sp in spans:
        out.append(text[cursor : sp["start"]])
        out.append(f"[{sp['label']}]")
        cursor = sp["end"]
    out.append(text[cursor:])
    return "".join(out), spans


# Model loading
def load_model(args, device, need_adapter):
    log.info(f"Loading base model {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=pick_dtype(),
        device_map={"": device},
        trust_remote_code=True,
    )

    is_peft = False
    if need_adapter:
        if args.adapter_path is None:
            raise RuntimeError("--adapter_path is required for the finetuned system")
        log.info(f"Attaching adapter {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        is_peft = True

        # Merging for a faster model - worked for SNIPER
        model = model.merge_and_unload()
        is_peft = False
        log.info("Adapter merged into the base weights")
    else:
        log.info("No adapter attached, this is the untouched pre-trained model")

    model.eval()
    model.config.use_cache = True
    return model, is_peft


# Generation
def generate(model, tkr, rows, shots, system_prompt, args, device):
    texts = []
    latencies = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc="generate"):
        chunk = rows[start : start + args.batch_size]
        batch = batch_prompts(tkr, chunk, shots, system_prompt, device)
        prompt_len = batch["input_ids"].shape[1]

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

        gen = tkr.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)
        texts.extend(t.strip() for t in gen)
        latencies.extend([1000.0 * (e - s) / len(chunk)] * len(chunk))

    return texts, latencies


# Systems
def run_system(name, rows, args, model, tkr, device):
    if name == "presidio":
        raise NotImplementedError(
            "Presidio is not implemented. The PRD allows either Presidio or the same "
            "Qwen2.5-1.5B-Instruct model zero-shot or 5-shot as the second baseline, and "
            "this submission uses the 5-shot model. Run --system fewshot instead."
        )

    if name == "regex":
        records = []
        for row in tqdm(rows, desc="regex"):
            s = time.time()
            text, _ = regex_sanitize(row["source_text"])
            e = time.time()
            records.append(make_record(row, text, latency_ms=1000.0 * (e - s)))
        return records

    if name == "finetuned":
        shots, prompt = [], SYSTEM_PROMPT
    elif name == "zeroshot":
        shots, prompt = [], ZEROSHOT_SYSTEM_PROMPT
    elif name == "fewshot":
        shots = pick_shots(args.train_file, args.num_shots, args.shot_max_chars)
        prompt = ZEROSHOT_SYSTEM_PROMPT
    else:
        raise NotImplementedError(f"{name} is not a valid system")

    # adapter disabled for the base model variants
    if (
        name in ("zeroshot", "fewshot")
        and getattr(model, "disable_adapter", None) is not None
    ):
        log.info("Disabling the adapter for this baseline")
        with model.disable_adapter():
            texts, latencies = generate(model, tkr, rows, shots, prompt, args, device)
    else:
        texts, latencies = generate(model, tkr, rows, shots, prompt, args, device)

    return [
        make_record(r, t, latency_ms=ms) for r, t, ms in zip(rows, texts, latencies)
    ]


# Split detection
def detect_split(test_file, args):
    if args.split != "auto":
        return args.split
    return "ood" if "ood" in os.path.basename(test_file).lower() else "in_distribution"


# Eval
def build_and_save(name, records, args, report_path, test_file, label_map):
    adapter = args.adapter_path if name == "finetuned" else "none"
    report = build_report(
        records, name, adapter, os.path.basename(test_file), label_map=label_map
    )
    save_json(report, report_path)
    log.info(f"📁 Report saved to {report_path}")
    log_report(report)
    return report


# Reporting
def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def log_report(report):
    log.info(f"n: {report['n']}")
    for key in ["exact", "overlap_iou50", "detection_iou50"]:
        if key in report:
            m = report[key]
            log.info(
                f"{key:22s} P {m['micro_p']:.4f}  R {m['micro_r']:.4f}  "
                f"F1 {m['micro_f1']:.4f}  macroF1 {m['macro_f1']:.4f}"
            )
    log.info(f"leak_rate: {report['leak_rate']:.4f}")
    log.info(f"over_masking_rate: {report['over_masking_rate']:.4f}")
    log.info(f"well_formed_rate: {report['well_formed_rate']:.4f}")

    # Per class table for the 10 most frequent classes
    log.info(f"{'class':<16}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
    for label, st in report["per_class"].items():
        log.info(
            f"{label:<16}{st['p']:>8.3f}{st['r']:>8.3f}{st['f1']:>8.3f}{st['support']:>9d}"
        )


# Table building
PARTB_CSV_COLUMNS = [
    "timestamp",
    "system",
    "split",
    "labels_converted",
    "test_set",
    "adapter",
    "n",
    "exact_micro_p",
    "exact_micro_r",
    "exact_micro_f1",
    "exact_macro_f1",
    "iou50_micro_p",
    "iou50_micro_r",
    "iou50_micro_f1",
    "iou50_macro_f1",
    "detection_micro_f1",
    "leak_rate",
    "over_masking_rate",
    "well_formed_rate",
]


def append_partb_row(report, split, converted, path):
    # Appends one row per system per split
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        existing = list(pd.read_csv(path, nrows=0).columns)
        if existing != PARTB_CSV_COLUMNS:
            raise RuntimeError(
                f"{path} was written with a different column schema. Move it aside and "
                f"let this run recreate it.\n  on disk: {existing}\n  expected: "
                f"{PARTB_CSV_COLUMNS}"
            )

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": report["system"],
        "split": split,
        "labels_converted": converted,
        "test_set": report["test_set"],
        "adapter": report["adapter"],
        "n": report["n"],
        "exact_micro_p": report["exact"]["micro_p"],
        "exact_micro_r": report["exact"]["micro_r"],
        "exact_micro_f1": report["exact"]["micro_f1"],
        "exact_macro_f1": report["exact"]["macro_f1"],
        "iou50_micro_p": report["overlap_iou50"]["micro_p"],
        "iou50_micro_r": report["overlap_iou50"]["micro_r"],
        "iou50_micro_f1": report["overlap_iou50"]["micro_f1"],
        "iou50_macro_f1": report["overlap_iou50"]["macro_f1"],
        "detection_micro_f1": report["detection_iou50"]["micro_f1"],
        "leak_rate": report["leak_rate"],
        "over_masking_rate": report["over_masking_rate"],
        "well_formed_rate": report["well_formed_rate"],
    }

    # Reindexed onto the fixed column list so every appended row lines up with the header
    df = pd.DataFrame([row]).reindex(columns=PARTB_CSV_COLUMNS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    log.info(
        f"📁 Appended {report['system']} ({split}, converted={converted}) to {path}"
    )


def main():
    args = get_args()

    log.info(f"{'-'*35} ⚙️ CONFIG {'-'*35}")
    for k in vars(args):
        log.info(f"{k}: {vars(args)[k]}")

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Data
    rows = load_jsonl(args.test_file)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    split = detect_split(args.test_file, args)
    log.info(f"Test rows: {len(rows)} from {args.test_file} | split: {split}")

    if args.convert_pii_labels and split == "in_distribution":
        log.info("⚠️ Label conversion on an in-distribution split is a no-op")

    # Model, only loaded for the systems that need one
    model, tkr = None, None
    if args.system in MODEL_SYSTEMS:
        tkr = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tkr.pad_token is None:
            tkr.pad_token = tkr.eos_token
        model, _ = load_model(args, device, need_adapter=args.system == "finetuned")

    # Eval
    log.info(f"{'-'*30} 🧪 SYSTEM: {args.system} {'-'*30}")
    records = run_system(args.system, rows, args, model, tkr, device)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_predictions)), exist_ok=True
    )
    write_predictions(records, args.output_predictions)
    log.info(f"📁 Predictions saved to {args.output_predictions}")

    if args.convert_pii_labels:
        log.info("Computing metrics with the OOD label mapping applied")
    label_map = OOD_LABEL_MAP if args.convert_pii_labels else None

    report = build_and_save(
        args.system, records, args, args.report_file, args.test_file, label_map
    )
    append_partb_row(report, split, args.convert_pii_labels, args.systems_csv)


if __name__ == "__main__":
    main()
