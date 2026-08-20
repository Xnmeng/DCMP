CUDA_VISIBLE_DEVICES=0 python retrieval_based_text_generation.py \
  --dataset_name cifar10 \
  --batch_size 64

CUDA_VISIBLE_DEVICES=0 python full_coprompt_owssl.py \
  --dataset_name cifar10 \
  --experiment_name cifar10_full_coprompt_seed1 \
  --seed 1 \
  --tau_u 0.1 \
  --tau_t_start 0.07 \
  --tau_t_end 0.04 \
  --output_dir exp \
  --backbone_name ViT-B/16 \
  --batch_size 64 \
  --epochs 200 \
  --n_ctx 4 \
  --prompt_depth 9 \
  --lambda_contrast 1.0 \
  --lambda_lcc_image 1.0 \
  --lambda_lcc_text 1.0
