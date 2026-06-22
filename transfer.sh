rsync -avh --progress \
  --exclude='.git/' \
  --exclude='data/' \
  --exclude='scope_saved/' \
  --exclude='new_outputs/' \
  --exclude='logs/' \
  --exclude='results/' \
  --exclude='outputs/' \
  --exclude='__pycache__/' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.bin' \
  --exclude='*.safetensors' \
  --exclude='*.ckpt' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.jsonl' \
  --exclude='wandb/' \
  /home/hyang/mediQ/ hyang@tesla.cs.ucla.edu:/home/hyang/mediQ

# rsync -avh --progress \
#   /home/hyang/mediQ/code-scope/code_feedback_cumulative_reward_mlp.pt \
#   /home/hyang/mediQ/scope_saved/transition_models/new/code_feedback_mdn_seed_0_batch_512 \
#   /home/hyang/mediQ/scope_saved/transition_models/new/code_feedback_mdn_seed_1_batch_512 \
#   /home/hyang/mediQ/scope_saved/transition_models/new/code_feedback_moe_seed_0_batch_2048 \
#   /home/hyang/mediQ/scope_saved/transition_models/new/code_feedback_moe_seed_1_batch_2048 \
#   /home/hyang/mediQ/scope_saved/transition_models/new/code_feedback_moe_seed_2_batch_2048 \
#   /home/hyang/mediQ/scope_saved/transition_models/new/code_feedback_moe_seed_3_batch_2048 \
#   hyang@tesla.cs.ucla.edu:/home/hyang/mediQ_model_files/