# python src/attack_extract.py \
# --victim_model Qwen/Qwen2.5-0.5B \
# --adapter_path adapters/victim_raw \
# --ref_model Qwen/Qwen2.5-0.5B \
# --corpus data/partC_corpus_raw.jsonl \
# --heldout data/partC_heldout.jsonl \
# --canaries data/partC_canaries.jsonl \
# --pii_index data/partC_pii_index.jsonl \
# --targeted_file data/partC_targeted.jsonl \
# --attack all \
# --num_samples 5000 \
# --seed 42 \
# --report_file results/partC_attacks_raw.json \
# --gpu_id 1
#
python src/attack_extract.py \
--victim_model Qwen/Qwen2.5-0.5B \
--adapter_path adapters/victim_sanitized \
--ref_model Qwen/Qwen2.5-0.5B \
--corpus data/partC_corpus_raw.jsonl \
--heldout data/partC_heldout.jsonl \
--canaries data/partC_canaries.jsonl \
--pii_index data/partC_pii_index.jsonl \
--targeted_file data/partC_targeted.jsonl \
--attack all \
--num_samples 5000 \
--seed 42 \
--report_file results/partC_attacks_sanitized.json \
--gpu_id 1

python src/attack_extract.py \
--victim_model Qwen/Qwen2.5-0.5B \
--ref_model Qwen/Qwen2.5-0.5B \
--corpus data/partC_corpus_raw.jsonl \
--heldout data/partC_heldout.jsonl \
--canaries data/partC_canaries.jsonl \
--pii_index data/partC_pii_index.jsonl \
--targeted_file data/partC_targeted.jsonl \
--attack all \
--num_samples 5000 \
--seed 42 \
--report_file results/partC_attacks_base.json \
--gpu_id 1
