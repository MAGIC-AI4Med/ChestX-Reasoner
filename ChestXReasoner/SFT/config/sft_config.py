from transformers import TrainingArguments
import os


class qwen2vl_sftconfig():
    def __init__(self):
        
        # transformers 下载库本地存储位置
        self.cache_dir="/mnt/lustre/fanziqing/radfm/checkpoint"   
        self.mode="reasoning"
        # self.mode="NOCOT"
        
        # transformer trainer中的 优化器，原始模型等存储路径, save="no"即不通过trainer保存
        # self.save_strategy="no"
        self.save_strategy="steps"
        self.save_steps=500
        self.save_only_model=True
        # self.output_dir="/mnt/lustre/fanziqing/radfm/radshare/RadFM/output/7Bsftmimic"
        self.output_dir="/mnt/lustre/fanziqing/radfm/radshare/RadFM_final/output/7Bcoldfinal"

        # transformers trainer 中的log和output文件路径, report_to="none"则不产生logging, Set to "wandb" if you are using Weights and Biases for logging
        # self.logging_dir="/mnt/lustre/fanziqing/radfm/radshare/log/sft_qwen2vl_nocot_llm" 
        # self.logging_dir="/mnt/lustre/fanziqing/radfm/radshare/log/sft_qwen2vl_combine_llm" 
        # self.logging_dir="/mnt/lustre/fanziqing/radfm/radshare/log/sft_qwen2vl_cot_7b_llmlora" 
        # self.logging_dir="/mnt/lustre/fanziqing/radfm/radshare/RadFM/log/sftmimic_7b"
        self.logging_dir="/mnt/lustre/fanziqing/radfm/radshare/RadFM_final/log/7bcoldfinal"


        # self.logging_dir="/mnt/lustre/fanziqing/radfm/radshare/log/sft_qwen2vl_combine_7b"
        self.report_to="tensorboard"
        self.logging_steps=10  # 多少step返回print一次loss
        self.logging_first_step=True
        self.evaluation_strategy="no"  # 是否在训练某个阶段做evaluation steps, epoch, no
        # self.evaluation_strategy="steps"
        self.eval_on_start=False
        # evaluation_strategy="steps",  # 每隔一定步数进行评估
        self.eval_steps=100  # 每500步评估一次
        # 自定义模型存储路径.最终模型和tokenizer存储路径
        # self.save_path="/mnt/lustre/fanziqing/radfm/radshare/savemodel/SFT_nocot_llm" 
        # self.save_path="/mnt/lustre/fanziqing/radfm/radshare/savemodel/SFT_combine_llm" 
        # self.save_path="/mnt/lustre/fanziqing/radfm/radshare/savemodel/SFT_cot_7blora" 
        # self.save_path="/mnt/lustre/fanziqing/radfm/radshare/savemodel/SFT_combine_7b" 
        # self.save_path="/mnt/lustre/fanziqing/radfm/radshare/RadFM/output/SFT_cot_7b"
        # self.save_path="/mnt/lustre/fanziqing/radfm/radshare/RadFM/output/SFT_mimic_7b"
        self.save_path="/mnt/lustre/fanziqing/radfm/radshare/RadFM_final/output/sftfinal"

        # self.save_path="/mnt/lustre/fanziqing/radfm/radshare/savemodel/SFT_cot_7b_llmlora" 
        self.llm_lora=False
        
        
        # sft 训练、测试json文件路径
        # self.train_datapath="/mnt/lustre/fanziqing/radfm/radshare/dataset/sft_10k_nocot_train.json"
        # self.test_datapath="/mnt/lustre/fanziqing/radfm/radshare/dataset/sft_10k_nocot_test.json"
        # self.train_datapath="/mnt/lustre/fanziqing/radfm/radshare/dataset/sft_10k_combine_train.json"
        # self.test_datapath="/mnt/lustre/fanziqing/radfm/radshare/dataset/sft_10k_combine_test.json"
        # self.train_datapath="/mnt/lustre/fanziqing/radfm/radshare/RadFM/dataset/sft_5k_nocot_train.json"
        # self.test_datapath="/mnt/lustre/fanziqing/radfm/radshare/RadFM/dataset/sft_5k_nocot_test.json"
        # self.train_datapath="/mnt/lustre/fanziqing/radfm/radshare/RadFM/dataset/binary_cls_train.json"
        # self.test_datapath="/mnt/lustre/fanziqing/radfm/radshare/RadFM/dataset/binary_cls_test.json"
        
        self.dataloader_num_workers=48
        # self.BATCH_SIZE=64
        self.BATCH_SIZE=128

        self.per_device_train_batch_size=4
        self.per_device_eval_batch_size=4
        # self.
        self.GRADIENT_ACCUMULATION_STEPS = self.BATCH_SIZE // self.per_device_train_batch_size

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            self.GRADIENT_ACCUMULATION_STEPS = self.GRADIENT_ACCUMULATION_STEPS // world_size

        self.num_train_epochs=5

        self.learning_rate=2e-6
        self.weight_decay=0.0
        self.warmup_ratio=0.1
        self.max_grad_norm=1.0
        self.lr_scheduler_type="cosine" 

        # self.deepspeed_config="/mnt/lustre/fanziqing/radfm/radshare/config/zero3_smaller.json"
        # self.deepspeed_config="/mnt/lustre/fanziqing/radfm/radshare/config/zero2.json"
        # self.deepspeed_config="/mnt/lustre/fanziqing/radfm/radshare/RadFM/config/zero3.json"
        # self.deepspeed_config="/mnt/lustre/fanziqing/radfm/radshare/RadFM_final/config/zero3_small.json"
        self.deepspeed_config="/mnt/lustre/fanziqing/radfm/radshare/RadFM_final/config/zero2.json"
        self.fp16=False  # Enable mixed precision for better performance on supported hardware
        self.bf16=True
        self.ddp_find_unused_parameters=False
        self.gradient_checkpointing=True

        ### 初始化huggingface的trainingaugments
        self.training_args=TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=self.evaluation_strategy,  # Evaluate at the end of each epoch
            eval_steps=self.eval_steps,
            logging_first_step=self.logging_first_step,
            eval_on_start=self.eval_on_start,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,  # 多少step返回一次loss
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            fp16=self.fp16,  # Enable mixed precision for better performance on supported hardware
            save_only_model=self.save_only_model,
            bf16=self.bf16,
            report_to=self.report_to,  # Set to "wandb" if you are using Weights and Biases for logging
            dataloader_num_workers=self.dataloader_num_workers,
            deepspeed=self.deepspeed_config,
            ddp_find_unused_parameters=self.ddp_find_unused_parameters,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type, 
            gradient_checkpointing=self.gradient_checkpointing
        )
        


if __name__=="__main__":
    training_args=qwen2vl_sftconfig()
    print(training_args.deepspeed_config)
    print(training_args.bf16)