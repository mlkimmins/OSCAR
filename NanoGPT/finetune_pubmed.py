import time

out_dir = 'out-pubmed'
# eval stuff
eval_interval = 100
eval_iters = 20
log_interval = 10
wandb_log = False # feel free to turn on
wandb_project = 'pubmed'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'pubmed'
init_from = 'gpt2-large' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 6
gradient_accumulation_steps = 5 * 8

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

max_iters = 6000



