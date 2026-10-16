#!/usr/bin/env bash
set -euo pipefail

MODEL=Qwen/Qwen2.5-1.5B-Instruct
ATTN=q_proj,k_proj,v_proj,o_proj
ATTN_MLP=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
HF_ADAPTER_REPO="bl0nder/llm-a1-adapters"

#==========================================================
# Download all trained adapters from Hugging Face
#==========================================================
python src/download_adapters.py \
--repo_id $HF_ADAPTER_REPO \
--adapter_path adapters

#==========================================================
# Part A: PEFT adapter placement sweep
#==========================================================
run_a () {
    # $1 run id, $2 target modules, $3 r, $4 alpha, $5 quant
    python src/peft_sweep.py \
    --model $MODEL --train_file data/partA_train.jsonl \
    --val_file data/partA_validation.jsonl \
    --target_modules $2 \
    --lora_r $3 --lora_alpha $4 \
    --lora_dropout 0.05 --quant $5 --seed 42 --run_id $1 \
    --output_dir adapters/$1 --results_csv results/partA_sweep.csv
}

run_a A1 q_proj,v_proj 8  16 none
run_a A2 $ATTN         8  16 none
run_a A3 $ATTN_MLP     8  16 none
run_a A4 $ATTN_MLP     32 64 none
run_a A5 $ATTN_MLP     32 16 none
run_a A6 $ATTN_MLP     32 64 4bit

# --make_fig only plots from the sweep CSV and does not train
python src/peft_sweep.py \
--results_csv results/partA_sweep.csv \
--make_fig

#==========================================================
# Part B: Sanitizer evaluation (best checkpoint A4)
#==========================================================
run_b () {
    # $1 test file, $2 system, $3 output suffix, remaining args passed through
    local test_file=$1 system=$2 suffix=$3
    shift 3
    python src/sanitize_eval.py \
    --base_model $MODEL \
    --adapter_path adapters/A4 \
    --test_file $test_file \
    --system $system \
    --output_predictions results/partB_predictions_${suffix}.jsonl \
    --report_file results/partB_metrics_${suffix}.json \
    "$@"
}

run_b data/partB_test.jsonl     finetuned finetuned
run_b data/partB_test.jsonl     regex     regex
run_b data/partB_test.jsonl     fewshot   fewshot
run_b data/partB_test.jsonl     zeroshot  zeroshot
run_b data/partB_ood_test.jsonl finetuned finetuned_ood
run_b data/partB_ood_test.jsonl finetuned finetuned_ood_converted --convert_pii_labels

#==========================================================
# Part C.1: Victim training
#==========================================================
python src/train_victim.py \
--model Qwen/Qwen2.5-0.5B \
--corpus data/partC_corpus_raw.jsonl \
--heldout data/partC_heldout.jsonl \
--target_modules $ATTN \
--lora_r 32 --lora_alpha 64 --epochs 1 --seed 42 \
--output_dir adapters/victim_raw

# Builds data/partC_corpus_sanitized.jsonl with adapters/A4 if it does not exist yet
python src/train_victim.py \
--model Qwen/Qwen2.5-0.5B \
--corpus data/partC_corpus_sanitized.jsonl \
--heldout data/partC_heldout.jsonl \
--target_modules $ATTN \
--lora_r 32 --lora_alpha 64 --epochs 1 --seed 42 \
--output_dir adapters/victim_sanitized

#==========================================================
# Part C.2: Extraction attacks (raw victim, sanitized victim, base model)
#==========================================================
run_c2 () {
    # $1 report suffix, remaining args passed through
    local suffix=$1
    shift
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
    --report_file results/partC_attacks_${suffix}.json \
    "$@"
}

run_c2 raw       --adapter_path adapters/victim_raw
run_c2 sanitized --adapter_path adapters/victim_sanitized
run_c2 base
