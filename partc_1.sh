# python src/train_victim.py \
# --model Qwen/Qwen2.5-0.5B \
# --corpus data/partC_corpus_raw.jsonl \
# --heldout data/partC_heldout.jsonl \
# --target_modules q_proj,k_proj,v_proj,o_proj \
# --lora_r 32 --lora_alpha 64 --epochs 1 --seed 42 \
# --output_dir adapters/victim_raw --gpu_id 1

python src/train_victim.py \
--model Qwen/Qwen2.5-0.5B \
--corpus data/partC_corpus_sanitized.jsonl \
--heldout data/partC_heldout.jsonl \
--target_modules q_proj,k_proj,v_proj,o_proj \
--lora_r 32 --lora_alpha 64 --epochs 1 --seed 42 \
--output_dir adapters/victim_sanitized --gpu_id 1
