CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
  --dataset_name cifar10 \
  --batch_size 64

CUDA_VISIBLE_DEVICES=0 python dcmp_owssl.py \
  --dataset_name cifar10 \
  --experiment_name cifar10 \
  --seed 1 \
  --tau_u 0.1 \
  --tau_t_start 0.07 \
  --tau_t_end 0.04 \
  --lambda_lcc 4.0 
 
