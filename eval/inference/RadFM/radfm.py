from typing import Any
import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.BLIP.multimodality_model import MultiLLaMAForCausalLM
import torch
import json
import os, sys
from utils import *
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image
from scipy import ndimage


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image prompt length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")
            image_padding_tokens.append(image_padding_token)
        text_tokenizer.add_special_tokens(
            special_token
        )
        # Ensure the bos, eos, and pad tokens are correctly set
        text_tokenizer.pad_token_id = 0
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens

def resize_image(image):
    target_D = 32
    if image.shape[-1] > target_D:
        image = ndimage.zoom(image, (3 / image.shape[0], 512 / image.shape[1], 512 / image.shape[2], target_D / image.shape[3]), order=0)
    else:
        image = ndimage.zoom(image, (3 / image.shape[0], 512 / image.shape[1], 512 / image.shape[2], 1), order=0)
    return image

def stack_images(images):
    
    target_H = 512
    target_W = 512
    target_D = 32
    if len(images) == 0:
        return torch.zeros((1,3,target_H,target_W,target_D))
    MAX_D = 32
    D_list = list(range(4,65,4))
    
    for ii in images:
        try:
            D = ii.shape[-1]
            if D > MAX_D:
                MAX_D = D
        except:
            continue
    for temp_D in D_list:
        if abs(temp_D - MAX_D)< abs(target_D - MAX_D):
            target_D = temp_D
    
    stack_images = []
    for s in images:
        # print(f'stack image function : s shape : {s.shape}')
        if len(s.shape) == 3:
        #print(s.shape)
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0).unsqueeze(-1), size = (target_H,target_W,target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0), size = (target_H,target_W,target_D)))
    images = torch.cat(stack_images, dim=0)
    return images

def combine_and_preprocess(question, image_list, image_padding_tokens):
    images = []
    new_question = question
    padding_index = 0
    for image in image_list:
        position = 0  # Modify as needed if position is different
        # Handle 2D images
        image = image.convert('RGB')
        transform = transforms.Compose([
            transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        image = transform(image)
        image = image.unsqueeze(-1)  # c, h, w, d
        image = image.float()
        images.append(image)
        # Add image placeholder to text
        image_placeholder = "<image>" + image_padding_tokens[padding_index] + "</image>"
        new_question = image_placeholder + new_question
        padding_index += 1
    vision_x = stack_images(images)  # Implement stack_images similar to code A
    
    vision_x = vision_x.unsqueeze(0)  # Now shape is [B, S, C, H, W, D]

    return new_question, vision_x

def main(task):
    with open("../inference/dataset.json", "r") as f:
        dataset_config = json.load(f)
    dataset_path = dataset_config[task]
    with open(dataset_path, "r") as f:
        data = json.load(f)
    # replace the path with your own checkpoint directory
    lang_encoder_path = "/path/to/your/model/checkpoint"
    tokenizer_path = '/path/to/your/model/checkpoint'
    checkpoint_path = '/path/to/your/model/checkpoint'


    text_tokenizer, image_padding_tokens = get_tokenizer(tokenizer_path)
    model = MultiLLaMAForCausalLM(
        lang_model_path=lang_encoder_path,  # Build up model based on LLaMa-13B config
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')  # Load model checkpoint
    # breakpoint()
    model.load_state_dict(ckpt)
    model = model.to('cuda')
    model.eval()
    print('RadFM model initialized')

    for item in tqdm(data):
        qs = item['question']+ " Let's think step by step, then answer the question."
        images = [Image.open(image).convert('RGB') for image in item['image_path']]
        text, vision_x = combine_and_preprocess(qs, images, image_padding_tokens)
        with torch.no_grad():
                lang_x = text_tokenizer(
                    text, max_length=2048, truncation=True, return_tensors="pt"
                )['input_ids'].to('cuda')

                vision_x = vision_x.to('cuda')
                generation = model.generate(lang_x, vision_x)
                outputs = text_tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
        item['pred']=outputs
        print(outputs)
    os.makedirs(f"../../output/radfm/", exist_ok=True)
    with open(f"../../output/radfm/{task}.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Run LLaVA-Next inference on a specific task.")
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate (e.g., mimic_binary, mimic_multi, etc.)")
    args = parser.parse_args()
    
    # 运行指定任务
    main(args.task)
