python src/peft_sweep.py \
--model Qwen/Qwen2.5-1.5B-Instruct --train_file data/partA_train.jsonl \
--val_file data/partA_validation.jsonl \
--target_modules q_proj,k_proj,v_proj,o_proj \
--lora_r 8 --lora_alpha 16 \
--lora_dropout 0.05 --quant none --seed 42 --run_id A2 \
--output_dir adapters/A2 --results_csv results/partA_sweep.csv \
--gpu_id 1 --make_fig
