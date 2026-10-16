import difflib
import json
import math
import re
import string
import zlib
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm

PII_LABELS = [
    "BOD",
    "BUILDING",
    "CARDISSUER",
    "CITY",
    "COUNTRY",
    "DATE",
    "DRIVERLICENSE",
    "EMAIL",
    "GEOCOORD",
    "GIVENNAME1",
    "GIVENNAME2",
    "IDCARD",
    "IP",
    "LASTNAME1",
    "LASTNAME2",
    "LASTNAME3",
    "PASS",
    "PASSPORT",
    "POSTCODE",
    "SECADDRESS",
    "SEX",
    "SOCIALNUMBER",
    "STATE",
    "STREET",
    "TEL",
    "TIME",
    "TITLE",
    "USERNAME",
]

# Matches a placeholder like [EMAIL] or [EMAIL_1] and captures the bare label
LABEL_RE = re.compile(r"\[([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*?)(?:_\d+)?\]")

# Label given to a region the model removed without leaving a placeholder behind
NO_LABEL = "O"

# OOD dataset uses a different vocab altogether wtf
# Mapping new labels to old ones (part A) to see if it's a skill issue or just a translation issue
OOD_LABEL_MAP = {
    "FIRSTNAME": "GIVENNAME1",
    "MIDDLENAME": "GIVENNAME2",
    "LASTNAME": "LASTNAME1",
    "PREFIX": "TITLE",
    "DOB": "BOD",
    "DATE": "DATE",
    "TIME": "TIME",
    "EMAIL": "EMAIL",
    "USERNAME": "USERNAME",
    "PASSWORD": "PASS",
    "IP": "IP",
    "IPV4": "IP",
    "IPV6": "IP",
    "PHONENUMBER": "TEL",
    "PHONEIMEI": "IDCARD",
    "SSN": "SOCIALNUMBER",
    "ACCOUNTNUMBER": "IDCARD",
    "CREDITCARDNUMBER": "IDCARD",
    "CREDITCARDISSUER": "CARDISSUER",
    "MASKEDNUMBER": "IDCARD",
    "STREET": "STREET",
    "STREETADDRESS": "STREET",
    "SECONDARYADDRESS": "SECADDRESS",
    "BUILDINGNUMBER": "BUILDING",
    "ZIPCODE": "POSTCODE",
    "CITY": "CITY",
    "STATE": "STATE",
    "COUNTY": "STATE",
    "NEARBYGPSCOORDINATE": "GEOCOORD",
    "SEX": "SEX",
    "GENDER": "SEX",
}


# Text normalization
def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def parse_label(chunk: str) -> str | None:
    m = LABEL_RE.search(chunk)
    if m is None:
        return None
    return m.group(1)


def split_placeholders(predicted: str) -> tuple[str, list[tuple[int, str]]]:
    literals = []
    holes = []
    cursor = 0
    lit_len = 0

    for m in LABEL_RE.finditer(predicted):
        chunk = predicted[cursor : m.start()]
        literals.append(chunk)
        lit_len += len(chunk)
        holes.append((lit_len, m.group(1)))
        cursor = m.end()

    literals.append(predicted[cursor:])
    return "".join(literals), holes


def align_literals(source: str, literals: str) -> list[int | None]:
    mapping = [None] * (len(literals) + 1)
    matcher = difflib.SequenceMatcher(None, source, literals, autojunk=False)

    for i, j, size in matcher.get_matching_blocks():
        for k in range(size):
            mapping[j + k] = i + k

    mapping[len(literals)] = len(source)
    return mapping


# Span recovery
def recover_spans(source: str, predicted: str) -> list[dict]:
    literals, holes = split_placeholders(predicted)
    if not holes:
        return []

    mapping = align_literals(source, literals)

    # Group placeholders that share the same literal boundary
    grouped = []
    for offset, label in holes:
        if grouped and grouped[-1][0] == offset:
            grouped[-1][1].append(label)
        else:
            grouped.append((offset, [label]))

    spans = []
    for offset, labels in grouped:
        start = walk_back(mapping, offset)
        end = walk_forward(mapping, offset, len(source))
        if end < start:
            continue

        # Consecutive placeholders share the region evenly
        width = (end - start) / len(labels)
        for n, label in enumerate(labels):
            lo = int(round(start + n * width))
            hi = int(round(start + (n + 1) * width))
            lo, hi = trim_region(source, lo, hi)
            if hi > lo:
                spans.append({"start": lo, "end": hi, "label": label})

    spans.sort(key=lambda s: (s["start"], s["end"]))
    return merge_adjacent(source, spans)


def walk_back(mapping: list[int | None], offset: int) -> int:
    # Source position just after the last literal character aligned before this placeholder
    for k in range(offset - 1, -1, -1):
        if mapping[k] is not None:
            return mapping[k] + 1
    return 0


def walk_forward(mapping: list[int | None], offset: int, default: int) -> int:
    # Source position of the first literal character aligned at or after this placeholder
    for k in range(offset, len(mapping)):
        if mapping[k] is not None:
            return mapping[k]
    return default


def trim_region(source: str, start: int, end: int) -> tuple[int, int]:
    junk = string.whitespace + string.punctuation
    while start < end and source[start] in junk:
        start += 1
    while end > start and source[end - 1] in junk:
        end -= 1
    return start, end


def merge_adjacent(source: str, spans: list[dict]) -> list[dict]:
    merged = []
    for span in spans:
        if merged:
            prev = merged[-1]
            gap = source[prev["end"] : span["start"]]
            if prev["label"] == span["label"] and gap.strip() == "":
                prev["end"] = span["end"]
                continue
        merged.append(dict(span))
    return merged


def drop_unlabelled(spans: list[dict]) -> list[dict]:
    return [s for s in spans if s["label"] != NO_LABEL]


def map_span_labels(spans: list[dict], mapping: dict) -> list[dict]:
    # Rewrites labels through a mapping. A label with no entry is left untouched so that it
    # still counts as a miss rather than silently disappearing.
    out = []
    for s in spans:
        t = dict(s)
        t["label"] = mapping.get(s["label"], s["label"])
        out.append(t)
    return out


# Span matching
def iou(a: dict, b: dict) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union


def match_exact(pred: list[dict], gold: list[dict]) -> list[tuple[int, int]]:
    # One to one, a prediction matches a gold span only on identical offsets and label
    pairs = []
    used_gold = set()
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if gi in used_gold:
                continue
            if (
                p["start"] == g["start"]
                and p["end"] == g["end"]
                and p["label"] == g["label"]
            ):
                pairs.append((pi, gi))
                used_gold.add(gi)
                break
    return pairs


def match_iou(
    pred: list[dict], gold: list[dict], thr: float = 0.5, ignore_label: bool = False
) -> list[tuple[int, int]]:
    # One to one, same label, greedy assignment in decreasing order of IoU
    candidates = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if not ignore_label and p["label"] != g["label"]:
                continue
            score = iou(p, g)
            if score >= thr:
                candidates.append((score, pi, gi))

    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    pairs = []
    used_pred = set()
    used_gold = set()
    for _, pi, gi in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        pairs.append((pi, gi))
    return pairs


# Entity metrics
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def entity_metrics(
    records: list[dict], mode: str = "exact", label_map: dict = None
) -> dict:
    # 'detection' ignores the label entirely so that span localization can be read apart from
    # naming, which is what makes the OOD numbers interpretable
    if mode == "exact":
        matcher = match_exact
    elif mode == "overlap_iou50":
        matcher = match_iou
    elif mode == "detection":
        matcher = lambda p, g: match_iou(p, g, ignore_label=True)
    else:
        raise NotImplementedError(f"{mode} is not a valid matching mode")

    tp_by_class = defaultdict(int)
    fp_by_class = defaultdict(int)
    fn_by_class = defaultdict(int)

    for rec in records:
        pred = drop_unlabelled(rec["predicted_spans"])
        gold = rec["gold_spans"]
        if label_map is not None:
            gold = map_span_labels(gold, label_map)
        pairs = matcher(pred, gold)

        matched_pred = {pi for pi, _ in pairs}
        matched_gold = {gi for _, gi in pairs}

        for _, gi in pairs:
            tp_by_class[gold[gi]["label"]] += 1
        for pi, p in enumerate(pred):
            if pi not in matched_pred:
                fp_by_class[p["label"]] += 1
        for gi, g in enumerate(gold):
            if gi not in matched_gold:
                fn_by_class[g["label"]] += 1

    tp = sum(tp_by_class.values())
    fp = sum(fp_by_class.values())
    fn = sum(fn_by_class.values())
    micro_p, micro_r, micro_f1 = prf(tp, fp, fn)

    # Per class table, macro averaged over classes that have gold support
    per_class = {}
    labels = set(tp_by_class) | set(fp_by_class) | set(fn_by_class)
    for label in sorted(labels):
        p, r, f1 = prf(tp_by_class[label], fp_by_class[label], fn_by_class[label])
        per_class[label] = {
            "p": p,
            "r": r,
            "f1": f1,
            "support": tp_by_class[label] + fn_by_class[label],
        }

    supported = [v["f1"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = sum(supported) / len(supported) if supported else 0.0

    return {
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def top_classes(per_class: dict, k: int = 10) -> dict:
    ranked = sorted(per_class.items(), key=lambda kv: -kv[1]["support"])
    return {label: stats for label, stats in ranked[:k]}


# Privacy metrics
def contains_value(hay: str, needle: str) -> bool:
    if len(needle) == 0:
        return False

    pattern = re.escape(needle)
    if needle[0].isalnum():
        pattern = r"(?<![0-9a-z])" + pattern
    if needle[-1].isalnum():
        pattern = pattern + r"(?![0-9a-z])"
    return re.search(pattern, hay) is not None


def leak_rate(records: list[dict]) -> float:
    # Fraction of gold PII values still present verbatim in the sanitized output
    total = 0
    leaked = 0
    for rec in records:
        hay = normalize(rec["predicted_text"])
        for value in rec["gold_values"]:
            needle = normalize(value)
            if len(needle) == 0:
                continue
            total += 1
            if contains_value(hay, needle):
                leaked += 1
    return leaked / total if total > 0 else 0.0


def leaked_values(record: dict) -> list[str]:
    hay = normalize(record["predicted_text"])
    return [v for v in record["gold_values"] if contains_value(hay, normalize(v))]


def over_masking_rate(records: list[dict]) -> float:
    # Fraction of non PII source characters that a replace or delete op touched
    total = 0
    masked = 0
    for rec in records:
        source = rec["source_text"]
        gold_chars = set()
        for g in rec["gold_spans"]:
            gold_chars.update(range(g["start"], g["end"]))

        touched = set()
        matcher = difflib.SequenceMatcher(
            None, source, rec["predicted_text"], autojunk=False
        )
        for tag, i1, i2, _, _ in matcher.get_opcodes():
            if tag in ("replace", "delete"):
                touched.update(range(i1, i2))

        for idx, ch in enumerate(source):
            if idx in gold_chars or ch.isspace():
                continue
            total += 1
            if idx in touched:
                masked += 1
    return masked / total if total > 0 else 0.0


def well_formed_rate(records: list[dict]) -> float:
    ok = 0
    for rec in records:
        text = rec["predicted_text"]
        if len(text.strip()) == 0:
            continue
        tokens = re.findall(r"\[([A-Z][A-Z0-9_]*)\]", text)
        if all(re.sub(r"_\d+$", "", t) in PII_LABELS for t in tokens):
            ok += 1
    return ok / len(records) if records else 0.0


# Report assembly
def head(m: dict) -> dict:
    return {
        "micro_p": m["micro_p"],
        "micro_r": m["micro_r"],
        "micro_f1": m["micro_f1"],
        "macro_f1": m["macro_f1"],
    }


def build_report(
    records: list[dict],
    system: str,
    adapter: str,
    test_set: str,
    label_map: dict = None,
) -> dict:
    exact = entity_metrics(records, "exact", label_map)
    overlap = entity_metrics(records, "overlap_iou50", label_map)
    detection = entity_metrics(records, mode="detection")

    report = {
        "system": system,
        "adapter": adapter,
        "test_set": test_set,
        "n": len(records),
        "exact": head(exact),
        "overlap_iou50": head(overlap),
        "per_class": top_classes(overlap["per_class"], k=10),
        "per_class_exact": top_classes(exact["per_class"], k=10),
        "leak_rate": leak_rate(records),
        "over_masking_rate": over_masking_rate(records),
        "well_formed_rate": well_formed_rate(records),
        "detection_iou50": head(detection),
    }

    return report


# Record construction
def make_record(row: dict, predicted_text: str, latency_ms: float = 0.0) -> dict:
    # Builds one results/partB_predictions.jsonl line from a raw dataset row and a generation
    gold_spans = [
        {"start": m["start"], "end": m["end"], "label": m["label"]}
        for m in row["privacy_mask"]
    ]
    rec = {
        "id": row["id"],
        "language": row.get("language", "English"),
        "source_text": row["source_text"],
        "predicted_text": predicted_text,
        "predicted_spans": recover_spans(row["source_text"], predicted_text),
        "gold_spans": gold_spans,
        "gold_values": [m["value"] for m in row["privacy_mask"]],
        "latency_ms": latency_ms,
    }
    rec["leaked_values"] = leaked_values(rec)
    return rec


def write_predictions(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Perplexity
def token_weighted_ppl(total_nll: float, total_tokens: int) -> float:
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


def corpus_perplexity(model, tkr, texts, device, max_len=512, batch_size=16):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for start in tqdm(range(0, len(texts), batch_size), desc="perplexity"):
        chunk = texts[start : start + batch_size]
        id_lists = [
            tkr(t, truncation=True, max_length=max_len)["input_ids"] for t in chunk
        ]
        id_lists = [ids for ids in id_lists if len(ids) >= 2]
        if not id_lists:
            continue

        width = max(len(ids) for ids in id_lists)
        input_ids = []
        labels = []
        attention_mask = []
        for ids in id_lists:
            pad = width - len(ids)
            input_ids.append(ids + [tkr.pad_token_id] * pad)
            labels.append(ids + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long, device=device
            ),
        }
        target = torch.tensor(labels, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(**batch).logits

        # Shift so position t predicts token t+1, then sum the NLL over real tokens only
        shift_logits = logits[:, :-1, :].float()
        shift_target = target[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_target.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )

        total_nll += nll.item()
        total_tokens += int((shift_target != -100).sum().item())

    return token_weighted_ppl(total_nll, total_tokens), total_nll, total_tokens


# Membership scores
def sequence_nll(model, tkr, texts, device, batch_size=32, max_len=512):
    model.eval()
    sums = [float("nan")] * len(texts)
    counts = [0] * len(texts)

    for start in tqdm(range(0, len(texts), batch_size), desc="nll"):
        chunk = texts[start : start + batch_size]
        id_lists = [
            tkr(t, truncation=True, max_length=max_len)["input_ids"] for t in chunk
        ]
        keep = [i for i, ids in enumerate(id_lists) if len(ids) >= 2]
        if not keep:
            continue

        width = max(len(id_lists[i]) for i in keep)
        input_ids = []
        labels = []
        attention_mask = []
        for i in keep:
            ids = id_lists[i]
            pad = width - len(ids)
            input_ids.append(ids + [tkr.pad_token_id] * pad)
            labels.append(ids + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long, device=device
            ),
        }
        target = torch.tensor(labels, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(**batch).logits

        # Per token NLL kept per row, so each text gets its own sum
        shift_logits = logits[:, :-1, :].float()
        shift_target = target[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_target.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_target.shape)

        row_sums = nll.sum(dim=1).tolist()
        row_counts = (shift_target != -100).sum(dim=1).tolist()
        for j, i in enumerate(keep):
            sums[start + i] = row_sums[j]
            counts[start + i] = int(row_counts[j])

    return sums, counts


def mean_nll(sums, counts):
    return [s / c if c > 0 else float("nan") for s, c in zip(sums, counts)]


def zlib_entropy(text):
    # Compressed bits per original byte, with the default zlib level as the PRD requires
    data = text.encode("utf-8")
    if len(data) == 0:
        return float("nan")
    return 8 * len(zlib.compress(data)) / len(data)


def membership_scores(nll_v, nll_ref, nll_lower, texts):
    # Appendix A. Every score is lower for sequences that look more like training members.
    scores = {"ppl": [], "ratio_ref": [], "ratio_lower": [], "ratio_zlib": []}
    for lv, lr, ll, text in zip(nll_v, nll_ref, nll_lower, texts):
        scores["ppl"].append(math.exp(lv))
        scores["ratio_ref"].append(math.exp(lv - lr))
        scores["ratio_lower"].append(math.exp(lv - ll))
        h = zlib_entropy(text)
        scores["ratio_zlib"].append(lv / h if h > 0 else float("nan"))
    return scores


def precision_at_k(scores, is_member, k=100):
    # Fraction of members among the k lowest scores
    ranked = sorted((s, i) for i, s in enumerate(scores) if s == s)
    top = [i for _, i in ranked[:k]]
    if not top:
        return 0.0
    return sum(1 for i in top if is_member[i]) / len(top)


# Discoverable memorization
def edit_similarity(a, b):
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


# Canary exposure
def canary_exposure(rank_sample, n_sampled=10000, space=10**6):
    rank_hat = rank_sample / n_sampled * space  # gotta use the raggedy ahh estimator
    return math.log2(space) - math.log2(rank_hat)


# Leakage
def value_leaked(text, value):
    return contains_value(normalize(text), normalize(value))
