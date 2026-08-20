

 CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
  --dataset_name flowers \
  --batch_size 64


CUDA_VISIBLE_DEVICES=0 python full_coprompt_owssl.py \
  --dataset_name flowers \
  --experiment_name flowers_lcc=2 \
  --batch_size 64 \
  --epochs 100 \
  --n_ctx 4 \
  --seed 4 \
  --prompt_depth 12 \
  --image_adapter_m 0.1 \
  --text_adapter_m 0.2 \
  --lambda_lcc_image 2 \
  --lambda_lcc_text 2 \
  --inference_mode soft