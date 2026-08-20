CUDA_VISIBLE_DEVICES=1 python retrieval_based_text_generation.py \
  --dataset_name cub \
  --batch_size 64


CUDA_VISIBLE_DEVICES=2 python dcmp_owssl.py \
  --dataset_name cub \
  --experiment_name cub \
  --batch_size 64 \
  --epochs 100 \
  --n_ctx 4 \
  --lambda_lcc 4 \
  --inference_mode soft 
