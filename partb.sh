python src/sanitize_eval.py \
--base_model Qwen/Qwen2.5-1.5B-Instruct \
--adapter_path adapters/A4 \
--test_file data/partB_test.jsonl \
--system finetuned \
--output_predictions results/partB_predictions_finetuned.jsonl \
--report_file results/partB_metrics_finetuned.json \
--gpu_id 2

python src/sanitize_eval.py \
--base_model Qwen/Qwen2.5-1.5B-Instruct \
--adapter_path adapters/A4 \
--test_file data/partB_test.jsonl \
--system regex \
--output_predictions results/partB_predictions_regex.jsonl \
--report_file results/partB_metrics_regex.json \
--gpu_id 2

python src/sanitize_eval.py \
--base_model Qwen/Qwen2.5-1.5B-Instruct \
--adapter_path adapters/A4 \
--test_file data/partB_test.jsonl \
--system fewshot \
--output_predictions results/partB_predictions_fewshot.jsonl \
--report_file results/partB_metrics_fewshot.json \
--gpu_id 2

python src/sanitize_eval.py \
--base_model Qwen/Qwen2.5-1.5B-Instruct \
--adapter_path adapters/A4 \
--test_file data/partB_test.jsonl \
--system zeroshot \
--output_predictions results/partB_predictions_zeroshot.jsonl \
--report_file results/partB_metrics_zeroshot.json \
--gpu_id 2

python src/sanitize_eval.py \
--base_model Qwen/Qwen2.5-1.5B-Instruct \
--adapter_path adapters/A4 \
--test_file data/partB_ood_test.jsonl \
--system finetuned \
--output_predictions results/partB_predictions_finetuned_ood.jsonl \
--report_file results/partB_metrics_finetuned_ood.json \
--gpu_id 2 

python src/sanitize_eval.py \
--base_model Qwen/Qwen2.5-1.5B-Instruct \
--adapter_path adapters/A4 \
--test_file data/partB_ood_test.jsonl \
--system finetuned \
--output_predictions results/partB_predictions_finetuned_ood_converted.jsonl \
--report_file results/partB_metrics_finetuned_ood_converted.json \
--gpu_id 2 --convert_pii_labels
