import argparse
import json
import os
import random
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from metrics import (
    canary_exposure,
    contains_value,
    corpus_perplexity,
    edit_similarity,
    mean_nll,
    membership_scores,
    normalize,
    precision_at_k,
    sequence_nll,
    value_leaked,
)
from peft_sweep import get_logger, load_jsonl, pick_dtype, set_seed

log = get_logger(__name__)

# Attack settings, fixed so every model is attacked in exactly the same way
PREFIX_TOKENS = 50
SUFFIX_TOKENS = 50
UNTARGETED_NEW_TOKENS = 128
TOP_K = 40
MATCH_WINDOW = 16
TARGETED_NEW_TOKENS = 32
TARGETED_SAMPLES = 10
CANARY_ALTERNATIVES = 9999
SECRET_SPACE = 10**6
CANARY_REPS = [1, 4, 16, 64]
ATTACKS = ["discoverable", "untargeted", "canary", "targeted"]

# Column order for the appendable attack table
CSV_COLUMNS = [
    "timestamp",
    "victim",
    "adapter",
    "attacks",
    "heldout_ppl",
    "disc_n_documents",
    "disc_n_excluded_short",
    "disc_exact_match_rate",
    "disc_mean_edit_sim",
    "untg_n_generated",
    "untg_n_unique",
    "untg_p100_ppl",
    "untg_p100_ratio_ref",
    "untg_p100_ratio_lower",
    "untg_p100_ratio_zlib",
    "untg_distinct_docs_recovered",
    "untg_distinct_pii_values_recovered",
    "untg_pii_recovery_rate",
    "canary_mean_exposure",
    "exposure_reps_1",
    "exposure_reps_4",
    "exposure_reps_16",
    "exposure_reps_64",
    "tgt_member_greedy_leak_rate",
    "tgt_member_sampled_any10_leak_rate",
    "tgt_control_greedy_leak_rate",
    "tgt_control_sampled_any10_leak_rate",
]


# Args
def get_args():
    parser = argparse.ArgumentParser(
        description="Part C.2 training data extraction attacks"
    )

    # Grading args
    parser.add_argument("--victim_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--ref_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", type=str, default="data/partC_corpus_raw.jsonl")
    parser.add_argument("--heldout", type=str, default="data/partC_heldout.jsonl")
    parser.add_argument("--canaries", type=str, default="data/partC_canaries.jsonl")
    parser.add_argument("--pii_index", type=str, default="data/partC_pii_index.jsonl")
    parser.add_argument(
        "--targeted_file", type=str, default="data/partC_targeted.jsonl"
    )
    parser.add_argument("--attack", type=str, default="all", choices=ATTACKS + ["all"])
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_file", type=str, default="results/partC_attacks.json")

    # Misc
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")

    return parser.parse_args()


# Model loading
def victim_name_from(adapter_path):
    if adapter_path is None or adapter_path.lower() == "none":
        return "base"
    name = os.path.basename(os.path.normpath(adapter_path)).lower()
    if "san" in name:
        return "victim_san"
    if "raw" in name:
        return "victim_raw"
    return name


def load_model(model_id, adapter_path, device):
    log.info(f"Loading {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=pick_dtype(),
        device_map={"": device},
        trust_remote_code=True,
    )
    # print(model)

    if adapter_path is not None and adapter_path.lower() != "none":
        log.info(f"Attaching adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model


def make_ref_scorer(args, model, tkr, device):
    if args.ref_model == args.victim_model and isinstance(model, PeftModel):

        def score(texts):
            with model.disable_adapter():
                return sequence_nll(model, tkr, texts, device, args.batch_size)

        return score

    if args.ref_model == args.victim_model:
        log.info("⚠️ The victim is the reference model, so s_ref is 1 for every sample")
        return lambda texts: sequence_nll(model, tkr, texts, device, args.batch_size)

    ref = load_model(args.ref_model, None, device)
    return lambda texts: sequence_nll(ref, tkr, texts, device, args.batch_size)


# Generation
def strip_after_eos(ids, eos_id):
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


def decode(
    model, tkr, prompts, max_new_tokens, device, batch_size, sample, num_return=1
):
    if sample:
        knobs = {"do_sample": True, "top_k": TOP_K, "top_p": 1.0, "temperature": 1.0}
    else:
        knobs = {"do_sample": False, "top_k": None, "top_p": None, "temperature": None}

    outputs = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        chunk = prompts[start : start + batch_size]
        width = max(len(p) for p in chunk)
        input_ids = [[tkr.pad_token_id] * (width - len(p)) + p for p in chunk]
        attention_mask = [[0] * (width - len(p)) + [1] * len(p) for p in chunk]
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long, device=device
            ),
        }

        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                num_return_sequences=num_return,
                pad_token_id=tkr.pad_token_id,
                eos_token_id=tkr.eos_token_id,
                **knobs,
            )

        for row in out[:, width:].tolist():
            outputs.append(strip_after_eos(row, tkr.eos_token_id))

    return outputs


# Data
def load_or_build_pii_index(args):
    if os.path.exists(args.pii_index):
        rows = load_jsonl(args.pii_index)
        log.info(f"Loaded {len(rows)} PII values from {args.pii_index}")
        return rows

    rows = []
    seen = set()
    for r in load_jsonl(args.targeted_file):
        if r["split"] == "member" and r["gold_value"] not in seen:
            seen.add(r["gold_value"])
            rows.append(
                {"value": r["gold_value"], "label": r["label"], "source": "targeted"}
            )
    for c in load_jsonl(args.canaries):
        if c["secret"] not in seen:
            seen.add(c["secret"])
            rows.append({"value": c["secret"], "label": "CANARY", "source": "canary"})

    os.makedirs(os.path.dirname(os.path.abspath(args.pii_index)), exist_ok=True)
    with open(args.pii_index, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"📁 Built PII index with {len(rows)} values at {args.pii_index}")
    return rows


# Held-out PPL
def heldout_perplexity(model, tkr, args, device):
    texts = [r["text"] for r in load_jsonl(args.heldout)]
    ppl, _, ntok = corpus_perplexity(model, tkr, texts, device, 512, args.batch_size)
    log.info(f"Held-out perplexity: {ppl:.4f} over {ntok} tokens")
    return ppl


# discoverable (a)
def attack_discoverable(model, tkr, corpus_rows, args, device):
    set_seed(args.seed)
    log.info(f"{'-'*30} 🔍 DISCOVERABLE MEMORIZATION {'-'*30}")

    # One entry per distinct document, repeats collapse onto their doc_id
    docs = {}
    for r in corpus_rows:
        if r["doc_id"] not in docs:
            docs[r["doc_id"]] = r

    prompts = []
    suffixes = []
    reps = []
    n_excluded = 0
    for r in docs.values():
        ids = tkr(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < PREFIX_TOKENS + SUFFIX_TOKENS:
            n_excluded += 1
            continue
        prompts.append(ids[:PREFIX_TOKENS])
        suffixes.append(ids[PREFIX_TOKENS : PREFIX_TOKENS + SUFFIX_TOKENS])
        reps.append(r["n_reps"])
    log.info(f"Documents scored: {len(prompts)} | excluded as too short: {n_excluded}")

    gens = decode(
        model, tkr, prompts, SUFFIX_TOKENS, device, args.batch_size, sample=False
    )

    buckets = {}
    exact = 0
    sims = []
    for gen, suffix, n in zip(gens, suffixes, reps):
        gen = gen[:SUFFIX_TOKENS]
        hit = gen == suffix
        exact += hit
        sims.append(edit_similarity(gen, suffix))
        bucket = buckets.setdefault(n, {"n": 0, "hits": 0})
        bucket["n"] += 1
        bucket["hits"] += hit

    by_n_reps = {}
    for n in sorted(buckets):
        by_n_reps[str(n)] = {
            "n": buckets[n]["n"],
            "exact_match_rate": buckets[n]["hits"] / buckets[n]["n"],
        }

    return {
        "n_documents": len(prompts),
        "n_excluded_short": n_excluded,
        "exact_match_rate": exact / len(prompts) if prompts else 0.0,
        "mean_edit_sim": sum(sims) / len(sims) if sims else 0.0,
        "by_n_reps": by_n_reps,
    }


# untargeted (b)
def build_window_index(tkr, corpus_rows):
    # Hashed MATCH_WINDOW token windows of every distinct raw document, mapped to its doc_id
    index = {}
    seen = set()
    for r in corpus_rows:
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        ids = tkr(r["text"], add_special_tokens=False)["input_ids"]
        for i in range(len(ids) - MATCH_WINDOW + 1):
            index.setdefault(hash(tuple(ids[i : i + MATCH_WINDOW])), r["doc_id"])
    return index


def matched_docs(ids, index):
    docs = set()
    for i in range(len(ids) - MATCH_WINDOW + 1):
        doc = index.get(hash(tuple(ids[i : i + MATCH_WINDOW])))
        if doc is not None:
            docs.add(doc)
    return docs


def attack_untargeted(model, tkr, ref_scorer, corpus_rows, pii_rows, args, device):
    set_seed(args.seed)
    log.info(f"{'-'*30} 🎣 UNTARGETED EXTRACTION {'-'*30}")

    # Unconditional samples from the start of text token
    start_id = tkr.bos_token_id if tkr.bos_token_id is not None else tkr.eos_token_id
    prompts = [[start_id] for _ in range(args.num_samples)]
    gens = decode(
        model, tkr, prompts, UNTARGETED_NEW_TOKENS, device, args.batch_size, sample=True
    )
    texts = [tkr.decode(g, skip_special_tokens=True).strip() for g in gens]

    # Duplicates ranked once otherwise a repeated sample would count twice
    unique = [t for t in dict.fromkeys(texts) if len(t) > 0]
    log.info(f"Generated {len(texts)} samples, {len(unique)} unique")

    # sample extracted if it shares a verbatim token window with a raw document
    index = build_window_index(tkr, corpus_rows)
    sample_docs = [
        matched_docs(tkr(t, add_special_tokens=False)["input_ids"], index)
        for t in unique
    ]
    is_member = [len(d) > 0 for d in sample_docs]
    log.info(f"Samples matching a training document: {sum(is_member)}")

    # membership scores
    nll_v = mean_nll(*sequence_nll(model, tkr, unique, device, args.batch_size))
    nll_ref = mean_nll(*ref_scorer(unique))
    lowered = [t.lower() for t in unique]
    nll_lower = mean_nll(*sequence_nll(model, tkr, lowered, device, args.batch_size))
    scores = membership_scores(nll_v, nll_ref, nll_lower, unique)

    precision = {}
    for name in ["ppl", "ratio_ref", "ratio_lower", "ratio_zlib"]:
        precision[name] = precision_at_k(scores[name], is_member, k=100)

    recovered_docs = set()
    for d in sample_docs:
        recovered_docs |= d

    hay = normalize(" | ".join(unique))
    recovered_values = [
        p["value"] for p in pii_rows if contains_value(hay, normalize(p["value"]))
    ]

    return {
        "n_generated": len(texts),
        "n_unique": len(unique),
        "precision_at_100": precision,
        "distinct_docs_recovered": len(recovered_docs),
        "distinct_pii_values_recovered": len(recovered_values),
        "pii_recovery_rate": len(recovered_values) / len(pii_rows) if pii_rows else 0.0,
    }


# canary (c)
def attack_canary(model, tkr, canaries, args, device):
    log.info(f"{'-'*30} 🐤 CANARY EXPOSURE {'-'*30}")
    rng = random.Random(args.seed)

    buckets = {}
    exposures = []
    at_ceiling = 0
    for c in canaries:
        true = int(c["secret"])
        alts = set()
        while len(alts) < CANARY_ALTERNATIVES:
            x = rng.randrange(SECRET_SPACE)
            if x != true:
                alts.add(x)

        width = len(c["secret"])
        candidates = [c["secret"]] + [f"{x:0{width}d}" for x in sorted(alts)]
        texts = [
            c["template"].format(project=c["project"], secret=s) for s in candidates
        ]

        sums, _ = sequence_nll(model, tkr, texts, device, args.batch_size * 16)
        rank = 1 + sum(1 for s in sums[1:] if s < sums[0])
        exposure = canary_exposure(rank, CANARY_ALTERNATIVES + 1, SECRET_SPACE)
        at_ceiling += rank == 1

        exposures.append(exposure)
        buckets.setdefault(c["n_reps"], []).append(exposure)
        log.info(
            f"{c['canary_id']} reps={c['n_reps']} rank={rank} exposure={exposure:.2f}"
        )

    log.info(
        f"Canaries at the estimator ceiling (rank 1): {at_ceiling} of {len(canaries)}"
    )

    by_n_reps = {}
    for n in sorted(buckets):
        by_n_reps[str(n)] = {
            "n": len(buckets[n]),
            "mean_exposure": sum(buckets[n]) / len(buckets[n]),
        }

    return {
        "n_canaries": len(canaries),
        "mean_exposure": sum(exposures) / len(exposures) if exposures else 0.0,
        "by_n_reps": by_n_reps,
    }


# targeted (d)
def attack_targeted(model, tkr, rows, args, device):
    set_seed(args.seed)
    log.info(f"{'-'*30} 🎯 TARGETED PII EXTRACTION {'-'*30}")

    prompts = [tkr(r["prompt"], add_special_tokens=False)["input_ids"] for r in rows]
    greedy = decode(
        model, tkr, prompts, TARGETED_NEW_TOKENS, device, args.batch_size, sample=False
    )
    sampled = decode(
        model,
        tkr,
        prompts,
        TARGETED_NEW_TOKENS,
        device,
        max(1, args.batch_size // TARGETED_SAMPLES),
        sample=True,
        num_return=TARGETED_SAMPLES,
    )

    result = {
        "n_member": sum(1 for r in rows if r["split"] == "member"),
        "n_control": sum(1 for r in rows if r["split"] == "control"),
    }
    for split in ["member", "control"]:
        idx = [i for i, r in enumerate(rows) if r["split"] == split]
        greedy_hits = 0
        sampled_hits = 0
        for i in idx:
            gold = rows[i]["gold_value"]
            if value_leaked(tkr.decode(greedy[i], skip_special_tokens=True), gold):
                greedy_hits += 1

            # Outputs come back grouped by prompt, TARGETED_SAMPLES in a row
            conts = sampled[i * TARGETED_SAMPLES : (i + 1) * TARGETED_SAMPLES]
            if any(
                value_leaked(tkr.decode(s, skip_special_tokens=True), gold)
                for s in conts
            ):
                sampled_hits += 1

        result[split] = {
            "greedy_leak_rate": greedy_hits / len(idx) if idx else 0.0,
            "sampled_any10_leak_rate": sampled_hits / len(idx) if idx else 0.0,
        }

    return result


# Reporting
def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log.info(f"📁 Report saved to {path}")


def build_row(args, report):
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "victim": report["victim"],
        "adapter": args.adapter_path or "none",
        "attacks": args.attack,
        "heldout_ppl": report["heldout_ppl"],
    }

    if "discoverable" in report:
        d = report["discoverable"]
        row["disc_n_documents"] = d["n_documents"]
        row["disc_n_excluded_short"] = d["n_excluded_short"]
        row["disc_exact_match_rate"] = d["exact_match_rate"]
        row["disc_mean_edit_sim"] = d["mean_edit_sim"]

    if "untargeted" in report:
        u = report["untargeted"]
        row["untg_n_generated"] = u["n_generated"]
        row["untg_n_unique"] = u["n_unique"]
        for name in ["ppl", "ratio_ref", "ratio_lower", "ratio_zlib"]:
            row[f"untg_p100_{name}"] = u["precision_at_100"][name]
        row["untg_distinct_docs_recovered"] = u["distinct_docs_recovered"]
        row["untg_distinct_pii_values_recovered"] = u["distinct_pii_values_recovered"]
        row["untg_pii_recovery_rate"] = u["pii_recovery_rate"]

    if "canary" in report:
        c = report["canary"]
        row["canary_mean_exposure"] = c["mean_exposure"]
        for n in CANARY_REPS:
            if str(n) in c["by_n_reps"]:
                row[f"exposure_reps_{n}"] = c["by_n_reps"][str(n)]["mean_exposure"]

    if "targeted" in report:
        t = report["targeted"]
        for split in ["member", "control"]:
            row[f"tgt_{split}_greedy_leak_rate"] = t[split]["greedy_leak_rate"]
            row[f"tgt_{split}_sampled_any10_leak_rate"] = t[split][
                "sampled_any10_leak_rate"
            ]

    return row


def append_attack_row(row, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Appending in frame order would silently misfile values under an older header
    if os.path.exists(path):
        existing = list(pd.read_csv(path, nrows=0).columns)
        if existing != CSV_COLUMNS:
            raise RuntimeError(
                f"{path} was written with a different column schema. Move it aside and "
                f"let this run recreate it."
            )

    df = pd.DataFrame([row]).reindex(columns=CSV_COLUMNS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    log.info(f"📁 Appended {row['victim']} to {path}")


def make_exposure_fig(csv_path, fig_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    cols = [f"exposure_reps_{n}" for n in CANARY_REPS]

    fig, ax = plt.subplots(figsize=(7, 5))
    drawn = 0

    # Latest canary run of each victim, both on the same axes
    for victim, style in [("victim_raw", "o-"), ("victim_san", "s--")]:
        rows = df[(df["victim"] == victim) & df["canary_mean_exposure"].notna()]
        if len(rows) == 0:
            continue
        latest = rows.iloc[-1]
        ax.plot(
            CANARY_REPS, [latest[c] for c in cols], style, label=victim, linewidth=2
        )
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return

    ceiling = canary_exposure(1, CANARY_ALTERNATIVES + 1, SECRET_SPACE)
    ax.axhline(
        ceiling,
        color="grey",
        linestyle=":",
        label=f"estimator ceiling ({ceiling:.2f} bits)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(CANARY_REPS)
    ax.set_xticklabels([str(n) for n in CANARY_REPS])
    ax.set_xlabel("Canary repetitions in training corpus")
    ax.set_ylabel("Mean exposure (bits)")
    ax.set_title("Canary exposure vs repetition count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"📁 Exposure figure saved to {fig_path}")


# Entry point
def main():
    args = get_args()

    log.info(f"{'-'*35} ⚙️ CONFIG {'-'*35}")
    for k in vars(args):
        log.info(f"{k}: {vars(args)[k]}")

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    victim = victim_name_from(args.adapter_path)
    attacks = ATTACKS if args.attack == "all" else [args.attack]
    log.info(f"Victim under attack: {victim} | attacks: {attacks}")

    # Tokenizer and model
    tkr = AutoTokenizer.from_pretrained(args.victim_model, trust_remote_code=True)
    if tkr.pad_token is None:
        tkr.pad_token = tkr.eos_token
    model = load_model(args.victim_model, args.adapter_path, device)

    report = {
        "victim": victim,
        "heldout_ppl": heldout_perplexity(model, tkr, args, device),
    }

    corpus_rows = None
    if "discoverable" in attacks or "untargeted" in attacks:
        corpus_rows = load_jsonl(args.corpus)

    # Attacking wooooo
    for name in attacks:
        # print(name)
        s = time.time()
        if name == "discoverable":
            report["discoverable"] = attack_discoverable(
                model, tkr, corpus_rows, args, device
            )
        elif name == "untargeted":
            ref_scorer = make_ref_scorer(args, model, tkr, device)
            pii_rows = load_or_build_pii_index(args)
            report["untargeted"] = attack_untargeted(
                model, tkr, ref_scorer, corpus_rows, pii_rows, args, device
            )
        elif name == "canary":
            report["canary"] = attack_canary(
                model, tkr, load_jsonl(args.canaries), args, device
            )
        elif name == "targeted":
            report["targeted"] = attack_targeted(
                model, tkr, load_jsonl(args.targeted_file), args, device
            )
        else:
            raise NotImplementedError(f"{name} is not a valid attack")
        e = time.time()
        log.info(f"⏰ {name} took {e - s:.1f} s")

    # Results
    save_json(report, args.report_file)
    results_dir = os.path.dirname(os.path.abspath(args.report_file))
    csv_path = os.path.join(results_dir, "partC_attacks.csv")
    append_attack_row(build_row(args, report), csv_path)
    if "canary" in report:
        make_exposure_fig(csv_path, os.path.join(results_dir, "partC_exposure.png"))


if __name__ == "__main__":
    main()
