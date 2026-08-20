python full_coprompt_owssl.py \
  --dataset_name cub \
  --experiment_name cub_xiugaixuexilv-2026-03-20-11-17 \
  --batch_size 64 \
  --epochs 100 \
  --prompt_depth 12 \
  --inference_mode soft \
  --image_vote_weight 0.5 \
  --eval_only \
  --resume exp/cub_xiugaixuexilv-2026-03-20-11-17/models/best_model.pth
