CUDA_VISIBLE_DEVICES=1 python retrieval_based_text_generation.py \
  --dataset_name cifar100 \
  --batch_size 64


CUDA_VISIBLE_DEVICES=1 python dcmp_owssl.py \
  --dataset_name cifar100 \
  --experiment_name cifar100_seed4 \
  --batch_size 64 \
  --epochs 100 \
  --n_ctx 4 \
  --seed 4 \
  --lambda_lcc 4.0 \
  --inference_mode soft