import argparse
import json
import math
import torch
from transformers import LlamaTokenizer
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import sys
import os

# Import MedDr model and image processor
from src.model.internvl_chat import InternVLChatModel, InternVLChatConfig
from src.dataset.transforms import build_transform
import pandas as pd

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
DEFAULT_IMAGE_TOKEN='<image>'
def load_model(model_name, device_map):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = InternVLChatModel.from_pretrained(
        model_name, 
        device_map=device_map,  # 自动将模型分布到可用GPU
        torch_dtype=torch.bfloat16,
    ).eval()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    pad2square = model.config.pad2square
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    
    image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)

    return model, tokenizer, image_processor

def split_model_internvl():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 60
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 60  # 从模型配置中确认的层数
    
    # 为GPU 0分配较少的层，因为它还需要处理视觉模型
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    
    # 添加所有可能的层路径变体
    layer_patterns = [
        'language_model.model.layers.{}',
        'language_model.base_model.model.layers.{}',
        'language_model.base_model.model.model.layers.{}'
    ]
    
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            if layer_cnt < num_layers:
                # 为每个层添加所有可能的路径变体
                for pattern in layer_patterns:
                    device_map[pattern.format(layer_cnt)] = i
                layer_cnt += 1
    
    # 添加视觉模型和连接层
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    
    # 特别添加所有可能的lm_head路径变体 - 修复当前错误
    lm_head_paths = [
        'language_model.lm_head',
        'language_model.model.lm_head',
        'language_model.base_model.lm_head',
        'language_model.base_model.model.lm_head',  # 这是错误消息中的路径
        'language_model.base_model.model.model.lm_head'
    ]
    
    for path in lm_head_paths:
        device_map[path] = 0
    
    # 添加所有其他组件的路径变体
    components = {
        'embed_tokens': [
            'language_model.model.embed_tokens',
            'language_model.model.tok_embeddings',
            'language_model.base_model.model.embed_tokens',
            'language_model.base_model.model.tok_embeddings',
            'language_model.base_model.model.model.embed_tokens',
            'language_model.base_model.model.model.tok_embeddings'
        ],
        'norm': [
            'language_model.model.norm',
            'language_model.base_model.model.norm',
            'language_model.base_model.model.model.norm'
        ],
        'output': [
            'language_model.output',
            'language_model.base_model.output'
        ],
        'rotary_emb': [
            'language_model.model.rotary_emb',
            'language_model.base_model.model.rotary_emb',
            'language_model.base_model.model.model.rotary_emb'
        ]
    }
    
    # 将所有组件路径分配到GPU 0
    for component_type, paths in components.items():
        for path in paths:
            device_map[path] = 0
    
    # 确保最后一层在GPU 0上
    for pattern in layer_patterns:
        device_map[pattern.format(num_layers - 1)] = 0
    
    return device_map

def main(task):
    with open("../inference/dataset.json", "r") as f:
        dataset_config = json.load(f)
    dataset_path = dataset_config[task]
    with open(dataset_path, "r") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = split_model()
    print(f"Device: {device}")  
    # Replace the string with your meddr checkpoint directory
    model, tokenizer, image_processor = load_model("/path/to/your/model/checkpoint", device_map)
    
    for item in tqdm(data):
        qs = item['question']
        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(item['image_path'])
        image_token_str = ' '.join(image_tokens)
        qs = f"{image_token_str} {qs} Let's think step by step, then answer the question."
        images = [Image.open(image).convert('RGB') for image in item['image_path']]
        image_tensor = [image_processor(image).unsqueeze(0).to(device).to(torch.bfloat16) for image in images]

        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=image_tensor,
                question=qs,
                generation_config={
                    "num_beams": 5,
                    "max_new_tokens": 128,
                    "do_sample": False,
                },
                print_out=False
            )
        item['pred']=response
    os.makedirs(f"../../output/meddr/", exist_ok=True)
    with open(f"../../output/meddr/{task}.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Run LLaVA-Next inference on a specific task.")
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate (e.g., mimic_binary, mimic_multi, etc.)")
    args = parser.parse_args()
    
    # 运行指定任务
    main(args.task)