 CUDA_VISIBLE_DEVICES=2 python retrieval_based_text_generation.py \
  --dataset_name scars \
  --batch_size 64


CUDA_VISIBLE_DEVICES=2 python dcmp_owssl.py \
  --dataset_name scars \
  --experiment_name scars \
  --batch_size 64 \
  --epochs 50 \
  --n_ctx 4 \
  --seed 4 \
  --lambda_lcc 8 \
  --inference_mode soft