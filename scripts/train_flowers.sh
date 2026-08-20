

 CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
  --dataset_name flowers \
  --batch_size 64


CUDA_VISIBLE_DEVICES=0 python dcmp_owssl.py \
  --dataset_name flowers \
  --experiment_name flowers \
  --batch_size 64 \
  --epochs 100 \
  --n_ctx 4 \
  --seed 4 \
  --image_adapter_m 0.1 \
  --text_adapter_m 0.2 \
  --lambda_lcc 4.0 \
  --inference_mode soft