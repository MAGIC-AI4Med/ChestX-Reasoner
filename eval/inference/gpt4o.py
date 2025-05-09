import os, sys
import json
from tqdm import tqdm
from copy import deepcopy
import argparse
from PIL import Image
import io
import base64
from openai import OpenAI

def query(input_messages) -> str:
    try:
        client = OpenAI(
            base_url="YOUR-BASE-URL-HERE",
            api_key="YOUR-API-KEY-HERE"
        )
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=input_messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in query: {e}")
        return ""

def encode_pil_image_to_base64(image: str) -> str:
    buffered = io.BytesIO()
    image = Image.open(image)
    if image.mode in ['F', 'CMYK']:
        image = image.convert('RGB')
    image.save(buffered, format="png")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_string

def main(task):
    with open("./inference/dataset.json","r") as f:
        dataset_config = json.load(f)
    dataset_path = dataset_config[task]
    with open(dataset_path, "r") as f:
        data = json.load(f)
    data = data
    for item in tqdm(data, desc=f"Processing {task}"):
        question = item['question']
        image_paths = item['image_path']
        
        content = [
            {
                "type": "text",
                "text": f"{question} Let's think step by step, then answer the question."
            }
        ]
        
        for img_path in image_paths:
            encoded_image = encode_pil_image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
            })
        
        input_message = [
            {
                "role": "system",
                "content": "You are a medical AI assistant, please answer the question step by step."
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        answer = query(input_message)
        item['pred'] = answer
        print(answer)
        
    # 创建输出目录
    output_dir = "../output/gpt4o/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    output_path = f"{output_dir}/{task}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Run GPT-4o inference on a specific task.")
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate (e.g., mimic_binary, mimic_multi, etc.)")
    args = parser.parse_args()
    
    # 运行指定任务
    main(args.task)