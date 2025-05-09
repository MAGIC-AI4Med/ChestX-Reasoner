import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
# from datasets import Dataset
from transformers import AutoProcessor
from collections import defaultdict
import random
from .prompt import sft_format_nocot_prompt, sft_format_cot_prompt,sft_system_cot_prompt,sft_system_nocot_prompt,inference_prompt
final_prompt=r"Let's think step by step. Enclose the reasoning within <think></think> and the answer within <answer></answer>."
class SFT_CombinedDataset(Dataset):
    def __init__(self, datasets, sampling_probs):
        self.datasets = datasets
        self.sampling_probs = sampling_probs
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 根据采样概率选择一个数据集
        dataset_idx = torch.multinomial(torch.tensor(self.sampling_probs), 1).item()
        # 从选定的数据集中随机选择一个样本
        sample_idx = torch.randint(0, self.lengths[dataset_idx], (1,)).item()
        return self.datasets[dataset_idx][sample_idx]


class SFT_Dataset(Dataset):
    def __init__(self, datapath: str) -> None:
        super().__init__()
        self.samples=self.build_dataset(datapath)
        
    def build_dataset(self, datapath: str):
        with open(datapath, 'r') as f:
            samples = json.load(f)
        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample=self.samples[index]
        item_id=sample["item_id"]
        image_path_list=sample["image_path"]
        question=sample["question"]
        answer=sample["answer"]
        dataset_name=sample["dataset_name"]

        if "<answer>" not in sample["answer"] or "</answer>" not in sample["answer"]:
            answer="<answer> "+sample["answer"]+" </answer>"
        
        
        if "process" in sample:
            if sample["process"] is not None:
                dataset_name="reasoning"
                process=sample["process"]
                if "<think>" not in process or "</think>" not in process:
                    process="<think> "+sample["process"]+" </think>"

                answer=process+answer

        return dataset_name,item_id,question,answer,image_path_list

class Evaluation_Dataset(Dataset):
    def __init__(self, datapath: str) -> None:
        super().__init__()
        self.samples=self.build_dataset(datapath)
        
    def build_dataset(self, datapath: str):
        with open(datapath, 'r') as f:
            samples = json.load(f)
        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample=self.samples[index]
        item_id=sample["item_id"]
        image_path_list=sample["image_path"]
        question=sample["question"]
        answer=sample["answer"]
        dataset_name=sample["dataset_name"]
        original_report = sample["original_report"]
        if "<answer>" not in sample["answer"] or "</answer>" not in sample["answer"]:
            answer="<answer> "+sample["answer"]+" </answer>"
        
        
        if "process" in sample:
            if sample["process"] is not None:
                
                process=sample["process"]
                if "<think>" not in process or "</think>" not in process:
                    process="<think> "+sample["process"]+" </think>"

                answer=process+answer

        return dataset_name,item_id,question,answer,image_path_list, original_report




def Qwen2vl_build_qaimage(dataset_name, processor, question, answer, image_path_list,mode="COT"):
    if mode=="COT":
        SYSTEM_PROMPT=sft_system_cot_prompt
        FORMAT_PROMPT=sft_format_cot_prompt
    elif mode=="NOCOT":
        SYSTEM_PROMPT=sft_system_nocot_prompt
        FORMAT_PROMPT=sft_format_nocot_prompt
    elif mode=="reasoning":
        SYSTEM_PROMPT=sft_system_cot_prompt
        FORMAT_PROMPT=sft_format_cot_prompt
    elif mode=="inference":
        SYSTEM_PROMPT=inference_prompt
        FORMAT_PROMPT=''
    img_placeholder={"type": "image"}  
    question_image=[img_placeholder for i in range(len(image_path_list))]
    # print(dataset_name)
    if "reasoning" in dataset_name:
        question_txt={"type": "text", "text": question+"\n"+final_prompt}
    else:
        question_txt={"type": "text", "text": question}

    question_image.append(question_txt)

    messages = [
        {"role": "user", "content": question_image},
    ]


    prefix_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,add_special_tokens=True,
    )
    # print(prefix_text)
    # print(answer)
    # print()

    image_list=[]
    for image_path in image_path_list:
        
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((448, 448))
        image_list.append(image)

    inputs = processor(text=prefix_text, images=image_list, return_tensors="pt") 

    
    answer_input_ids = processor.tokenizer.encode(
        answer,
        return_tensors="pt",
        # max_length=4096,
        # add_special_tokens=True,
        padding='longest',
        truncation=True,
    )

    return dict(
        q_input_ids=inputs["input_ids"],
        pixel_values= inputs["pixel_values"],
        a_input_ids=answer_input_ids,
        image_grid_thw=inputs["image_grid_thw"]
    )
    #'''


class Qwen2vl_TrainCollator:
    def __init__(self, processor, IGNORE_INDEX=-100,mode="NOCOT"):
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX
        self.mode=mode


    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,

    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        
        return input_ids, labels

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        new_batch = defaultdict(list)

        for feature in features:
            dataset_name,item_id,question,answer,image_path_list=feature
            qaimage_output = Qwen2vl_build_qaimage(
            dataset_name, self.processor, question, answer, image_path_list,self.mode
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output["q_input_ids"], qaimage_output["a_input_ids"]
            )
            new_batch["max_input_len_list"].append(temp_input_ids.shape[1])
            new_batch["input_ids_list"].append(temp_input_ids)
            new_batch["labels_list"].append(temp_labels)
            new_batch["pixel_values"].append(qaimage_output["pixel_values"])
            new_batch["image_grid_thw"].append(qaimage_output["image_grid_thw"])


        max_input_len = max(new_batch["max_input_len_list"])

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [   
                        torch.full(
                            (1, max_input_len - new_batch["max_input_len_list"][index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                        
                    ],
                    axis=1,
                )
                for index, value in enumerate(new_batch["input_ids_list"])
            ]
        )

        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - new_batch["max_input_len_list"][index]),
                            self.ingnore_index,
                        ),
                        value,
                        
                    ],
                    axis=1,
                )
                for index, value in enumerate(new_batch["labels_list"])
            ]
        )


        final_pixel_values = torch.concat(new_batch["pixel_values"], axis=0)
        image_grid_thw=torch.concat(new_batch["image_grid_thw"],axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return dict(
            input_ids= final_input_ids,
            labels= final_labels,
            pixel_values= final_pixel_values,
            attention_mask= attention_mask,
            image_grid_thw=image_grid_thw
        )
        
class Qwen2vl_TestCollator:
    def __init__(self, processor, IGNORE_INDEX=-100,mode="COT"):
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX
        self.mode=mode

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,

    ):
        input_ids=q_input_ids
        
        return input_ids

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        new_batch = defaultdict(list)

        for feature in features:
            dataset_name,item_id,question,answer,image_path_list=feature
            qaimage_output = Qwen2vl_build_qaimage(
            dataset_name, self.processor, question, answer, image_path_list,self.mode
            )
            temp_input_ids= self.convert_one_piece(
                qaimage_output["q_input_ids"]
            )
            new_batch["max_input_len_list"].append(temp_input_ids.shape[1])

            new_batch["input_ids_list"].append(temp_input_ids)

            new_batch["pixel_values"].append(qaimage_output["pixel_values"])
            new_batch["image_grid_thw"].append(qaimage_output["image_grid_thw"])


        max_input_len = max(new_batch["max_input_len_list"])

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [   
                        torch.full(
                            (1, max_input_len - new_batch["max_input_len_list"][index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                        
                    ],
                    axis=1,
                )
                for index, value in enumerate(new_batch["input_ids_list"])
            ]
        )
        

        final_pixel_values = torch.concat(new_batch["pixel_values"], axis=0)
        image_grid_thw=torch.concat(new_batch["image_grid_thw"],axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return dict(
            input_ids= final_input_ids,
            pixel_values= final_pixel_values,
            attention_mask= attention_mask,
            image_grid_thw=image_grid_thw
        )
