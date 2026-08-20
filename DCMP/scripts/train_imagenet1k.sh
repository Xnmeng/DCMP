CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
  --dataset_name imagenet_1k \
  --batch_size 64


CUDA_VISIBLE_DEVICES=0 python full_coprompt_owssl.py \
  --dataset_name imagenet_1k \
  --experiment_name imagenet1K_data \
  --batch_size 64 \
  --epochs 100 \
  --n_ctx 4 \
  --prompt_depth 12 \
  --image_adapter_m 0.1 \
  --text_adapter_m 0.2 \
  --lambda_lcc_image 4.0 \
  --lambda_lcc_text 4.0 \
  --inference_mode soft