python full_coprompt_owssl.py \
  --dataset_name scars \
  --experiment_name scars_seed4-2026-03-24-15-39 \
  --batch_size 64 \
  --epochs 100 \
  --prompt_depth 12 \
  --inference_mode soft \
  --image_vote_weight 0.5 \
  --eval_only \
  --resume exp/scars_seed4-2026-03-24-15-39/models/best_model.pth