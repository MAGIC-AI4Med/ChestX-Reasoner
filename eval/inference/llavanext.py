import torch
from PIL import Image
import json
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from copy import deepcopy
from transformers import set_seed, logging
import argparse

logging.set_verbosity_error()

def main(task):
    with open("./inference/dataset.json", "r") as f:
        dataset_config = json.load(f)
    dataset_path = dataset_config[task]
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    pretrained = "/path/to/your/model/checkpoint"
    model_name = "llava_llama3"
    device = "cuda"
    device_map = "auto"
    
    # 加载模型和处理器
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, 
        None, 
        model_name, 
        device_map=device_map
    )
    
    # 设置tokenizer到对话模板
    conv_templates["llava_llama_3"].tokenizer = tokenizer
    
    model.eval()
    model.tie_weights()

    for item in tqdm(data):
        question = item['question'] + " Let's think step by step, then answer the question."
        image_paths = item['image_path']
        images = [Image.open(image).convert('RGB') for image in image_paths]
        # 处理图片
        image_tensor = process_images(images, image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        
        # 构建对话
        conv = deepcopy(conv_templates["llava_llama_3"])
        question_with_image = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 准备输入
        input_ids = tokenizer_image_token(
            prompt, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).to(device)
        
        image_sizes = [image.size for image in images]
        
        # 生成回答
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )
        
        # 解码输出
        text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        item['pred'] = text_output
        print(text_output)

    with open(f'./output/llavanext/{task}.json', "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Run LLaVA-Next inference on a specific task.")
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate (e.g., mimic_binary, mimic_multi, etc.)")
    args = parser.parse_args()
    
    # 运行指定任务
    main(args.task)