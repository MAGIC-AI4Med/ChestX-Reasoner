## Instruction for supervised fine-tuning
Training parameters are set in ./sft_config.py. Most parameters are used for huggingface trainer. For example:  
```bash
cache_dir: your path to load base model (Qwen2VL-7B in our case)  
mode: output mode to load different prompt("reasoning","nocot"), you should manually set them in your dataloader  
save_strategy: whether to save model("no","steps","epoch")  
save_steps: step interval to save model, contradict with "no", and "epoch"(500)  
output_dir: your path to save ckpt
logging_dir: your path to save log file  
report_to: your way to log("tensorboard","wandb")  
logging_steps: how many training steps to show loss(10)  
logging_first_step: whether to log loss before the first step(True)
evaluation_strategy: way to evaluation("no","steps")  
eval_on_start: whether to perform evaluation in the first step(True, False)
eval_steps: 100 evaluation interval  
llm_lora: wheter to use lora(True, False)
dataloader_num_workers(64)  
BATCH_SIZE(128,256,512)  
per_device_train_batch_size(4)
per_device_eval_batch_size(4)  
num_train_epochs(5) total train epoch, can set larger than exact training steps to avoid too small lr  
learning_rate(2e-6)
weight_decay(0.0)  
warmup_ratio(0.1)  
max_grad_norm(1.0)  
lr_scheduler_type("cosine")   
deepspeed_config: your path to load deepspeed config("./config/zero2.json")
bf16(True)  
ddp_find_unused_parameters(False)  
gradient_checkpointing(True)  
```

## Instruction for RL tuning


