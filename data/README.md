# Assignment 1 dataset bundle

This directory is the frozen student-facing dataset bundle. Use the JSONL files as provided; do not download upstream datasets or create new splits. Each non-empty line is one JSON object.

## Files

- `partA_train.jsonl`: 20,000 English sanitizer-training examples.
- `partA_validation.jsonl`: 2,000 English model-selection examples.
- `partB_test.jsonl`: 2,000 held-out English sanitizer-test examples.
- `partB_ood_test.jsonl`: 1,000 English near-OOD examples from the independent ai4privacy 200K release.
- `partC_corpus_raw.jsonl`: 8,007 physical victim-training rows representing 5,000 distinct base documents, controlled duplicates, and canaries.
- `partC_heldout.jsonl`: 500 never-trained documents.
- `partC_canaries.jsonl`: secret and repetition metadata for 20 canaries. Do not use this file as training data.
- `partC_targeted.jsonl`: 200 member and 200 non-member attack/calibration prompts.
- `manifest.jsonl`: row count, byte count, and SHA-256 digest for each dataset file.

## Integrity check

From the directory containing `data/`:

```bash
python - <<'PY'
import hashlib, json, pathlib
d = pathlib.Path("data")
for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    item = json.loads(line)
    payload = (d / item["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    assert len(payload.splitlines()) == item["rows"]
print("dataset bundle OK")
PY
```

The upstream sources are `ai4privacy/pii-masking-300k` and, for the near-OOD file, `ai4privacy/pii-masking-200k`. The instructor-side `build_datasets.py` records the deterministic construction procedure (seed 42). Malformed upstream rows whose declared character offsets do not recover their declared values are excluded.
