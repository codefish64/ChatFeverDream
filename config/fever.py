# ================================
# feverdreamLLM — tiny shitpost GPT
# ~30M params, runs on M1 under 90MB after quantization
# ================================

# OUTPUT
out_dir = 'out-fever'
eval_interval = 500
eval_iters = 200
log_interval = 50
always_save_checkpoint = True
wandb_log = False

# DATASET
dataset = 'fever'          # points to data/fever/
vocab_size = 8000
block_size = 128           # context length, shorter for M1

# TRAINING
batch_size = 8
gradient_accumulation_steps = 4  # effective batch ~32
learning_rate = 3e-4
max_iters = 60000
lr_decay_iters = 60000
min_lr = 3e-5
warmup_iters = 100
beta2 = 0.99
dropout = 0.0

# MODEL (tiny GPT-2 style)
n_layer = 6
n_head = 6
n_embd = 384

# MACBOOK / M1 SETTINGS
device = 'mps'            # Apple GPU
backend = 'mps'
dtype = 'float16'
compile = False

# ================================
# Notes:
# - batch_size and block_size are reduced to prevent freezing on M1
# - gradient_accumulation_steps scales effective batch without extra memory
# - dropout=0.0 for small datasets to overfit faster
# - This config keeps model under 90MB after quantization
# ================================
