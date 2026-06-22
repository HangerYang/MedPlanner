cd /home/hyang/mediQ

/home/hyang/miniconda3/envs/scope/bin/python \
  scripts/build_code_feedback_turn_features.py \
  --rewards data/med_data/code_feedback_rewards.jsonl \
  --embedding-dir data/med_data/data/embeddings \
  --split-manifest data/med_data/code_feedback_split.json \
  --test-conversations 1000 \
  --seed 42 \
  --train-output data/med_data/code_feedback_turn_features.train.jsonl \
  --test-output data/med_data/code_feedback_turn_features.test.jsonl \
  --summary data/med_data/code_feedback_turn_features.summary.json

/home/hyang/miniconda3/envs/scope/bin/python \
  scripts/train_code_feedback_feature_reward.py \
  --train data/med_data/code_feedback_turn_features.train.jsonl \
  --test data/med_data/code_feedback_turn_features.test.jsonl \
  --target cumulative_reward \
  --output scope_saved/reward/code_feedback_cumulative_reward_mlp.pt \
  --metrics scope_saved/reward/code_feedback_cumulative_reward_mlp.metrics.json \
  --epochs 20 \
  --batch-size 4096 \
  --lr 0.001