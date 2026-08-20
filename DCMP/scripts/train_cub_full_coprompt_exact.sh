CUDA_VISIBLE_DEVICES=1 python retrieval_based_text_generation.py \
  --dataset_name cub \
  --batch_size 64


CUDA_VISIBLE_DEVICES=2 python full_coprompt_owssl.py \
  --dataset_name cub \
  --experiment_name cub_lcc=8 \
  --batch_size 64 \
  --epochs 100 \
  --n_ctx 4 \
  --prompt_depth 12 \
  --image_adapter_m 0.1 \
  --text_adapter_m 0.2 \
  --lambda_lcc_image 8 \
  --lambda_lcc_text 8 \
  --inference_mode soft 
