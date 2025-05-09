from dataset.SFT_dataloader import SFT_Dataset,Qwen2vl_TrainCollator,SFT_CombinedDataset

from transformers import AutoProcessor, Trainer, TrainingArguments,Qwen2VLForConditionalGeneration,AutoModelForCausalLM,set_seed
import torch
import os
import deepspeed
from config.sft_config import qwen2vl_sftconfig
from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import random
def set_seeds():
    seed=66
    rank=int(os.environ.get("LOCAL_RANK", 0))
    print("set seed for rank:",rank)
    set_seed(seed)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def prepare_dataset(data_list,weights):
    train_dataset_list=[]
    combined_test=None
    test_dataset_list=[]
    # if "reasoning" in data_list[0]:
    test_dataset_list=None
    for dataset_name in data_list:
        if dataset_name == your dataset name1 :
            train_file= your train file dir
            test_file=your train file dir or None
        elif dataset_name == your dataset name2 :
            train_file= your train file dir2
            test_file=your train file dir2 or None
      
        else:
            print(dataset_name)
            raise NotImplementedError 
        train_dataset=SFT_Dataset(train_file)
        
        train_dataset_list.append(train_dataset)

        if test_dataset_list is not None:
            test_dataset=SFT_Dataset(test_file)
            test_dataset_list.append(test_dataset)
    combined_train=SFT_CombinedDataset(train_dataset_list,weights)
    if test_dataset_list is not None:
        combined_test=SFT_CombinedDataset(test_dataset_list,weights)
    return combined_train,combined_test

##############################   定义路径和参数   ###################################
set_seeds()

config=qwen2vl_sftconfig()
training_args = config.training_args

##############################  加载数据集和数据处理器  ############################
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    # model_max_length=2048,
    cache_dir=config.cache_dir)

tokenizer=processor.tokenizer

dataset_names= your dataset name lists
weights=your weights

'''
example case:
dataset_names=[
# "binary_mimic","binary_chexpert","binary_rsna","binary_siim","temporal","multi_mimic","multi_chexpert","single_mimic","single_chexpert",\
# "binary_mimic_no","binary_chexpert_no","binary_rsna_no","binary_siim_no","temporal_no","multi_mimic_no","multi_chexpert_no",\
"mimic_binary_reasoning","mimic_multi_reasoning","mimic_single_reasoning","chexpert_binary_reasoning","chexpert_multi_reasoning","chexpert_single_reasoning","temporal_reasoning",\
"mimic_binary_reasoning_no","mimic_multi_reasoning_no","chexpert_binary_reasoning_no","chexpert_multi_reasoning_no","temporal_reasoning_no"
]
weights=[
# 0.3,0.3,0.2,0.2,0.2,0.3,0.3,0.2,0.3, \
# 0.3,0.3,0.2,0.2,0.2,0.3,0.3, \
0.1,0.1,0.1,0.1,0.1,0.1,0.05, \
0.1,0.1,0.1,0.1,0.05
]
'''

s_index=range(len(weights))
random.shuffle(s_index)
dataset_names=[dataset_names[iitem] for iitem in s_index]
weights=[weights[iitem] for iitem in s_index]
print(dataset_names)

train_dataset,test_dataset=prepare_dataset(dataset_names,weights)


processed_datasets={}
processed_datasets["train"]=train_dataset
processed_datasets["test"]=test_dataset

data_collator=Qwen2vl_TrainCollator(processor=processor,mode=config.mode)

################################   加载模型和训练器  ################################
# model_dir="/mnt/lustre/fanziqing/radfm/radshare/RadFM/output/SFT_cot_7b/pytorch_model.bin"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    cache_dir=config.cache_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
# checkpoint = torch.load(model_dir)
# 将加载的权重传入模型
# model.load_state_dict(checkpoint)

# whether to use lora
if config.llm_lora:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
        r=8,  # Rank of LoRA
        lora_alpha=32,  # Alpha scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers

        )

    model=get_peft_model(model, lora_config)
# model.model.print_trainable_parameters()

##############################    设置训练参数    #################################

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets['train'],
    eval_dataset=processed_datasets['test'],  # Replace with a validation set if available
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)
model.train()

trainer.train()
