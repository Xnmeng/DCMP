 CUDA_VISIBLE_DEVICES=2 python retrieval_based_text_generation.py \
  --dataset_name scars \
  --batch_size 64


CUDA_VISIBLE_DEVICES=2 python full_coprompt_owssl.py \
  --dataset_name scars \
  --experiment_name scars_image_only \
  --batch_size 64 \
  --epochs 50 \
  --n_ctx 4 \
  --seed 4 \
  --prompt_depth 12 \
  --image_adapter_m 0.1 \
  --text_adapter_m 0.2 \
  --lambda_lcc_image 1 \
  --lambda_lcc_text 0 \
  --inference_mode soft