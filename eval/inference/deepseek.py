import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
import os, sys
from transformers import set_seed
import argparse  # 新增：引入 argparse 模块
def main(task):
    with open("./inference/dataset.json","r") as f:
        dataset_config = json.load(f)
    dataset_path = dataset_config[task]
    with open(dataset_path, "r") as f:
        data = json.load(f)

    model_path = '/path/to/your/model/checkpoint'
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    for item in tqdm(data):
        question = item['question']
        image_paths = item['image_path']
        images = [Image.open(image).convert('RGB') for image in image_paths]
        num_images = len(images)
        image_tokens = " ".join(["<image_placeholder>"] * num_images)
        qs = f"{image_tokens} {question} Let's think step by step, then answer the question."
        conversation = [
            {
                "role":"User",
                "content":qs,
                "images":images,
            },
            {
                "role":"Assistant",
                "content" : ""
            }
        ]
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=images, force_batchify=True
        ).to(vl_gpt.device)
        # print('prepare_inputs: ', prepare_inputs)
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        # print('inputs_embeds: ', inputs_embeds)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(answer)
        item['pred'] = answer

    with open(f'../output/deepseek/{task}.json',"w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # 新增：命令行参数解析
    parser = argparse.ArgumentParser(description="Run DeepSeek-VL inference on a specific task.")
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate (e.g., mimic_binary, mimic_multi, etc.)")
    args = parser.parse_args()
    
    # 运行指定任务
    main(args.task)