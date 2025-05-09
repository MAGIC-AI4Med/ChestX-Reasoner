import os
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import json
from tqdm import tqdm
from itertools import islice
from qwen_vl_utils import process_vision_info
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
def batchify(data, batch_size):
    """将数据分成批量大小为 batch_size 的小批次，并打印实际批次信息"""
    print(f"数据总量: {len(data)}, 批次大小: {batch_size}")
    it = iter(data)
    # breakpoint()
    batches = list(iter(lambda: list(islice(it, batch_size)), []))  # 强制展开所有批次
    print(f"总批次数: {len(batches)}")
    for i, batch in enumerate(batches):
        print(f"批次 {i+1}: 样本数 {len(batch)}")
    return iter(batches)  # 返回迭代器以兼容原有代码

def main(task, batch_size=8):
    # 加载模型和处理器
    print(f"cuda visible device: {os.environ['CUDA_VISIBLE_DEVICES']}")
    cache_dir = "/path/to/your/model/checkpoint"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cache_dir, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        device_map="auto",
        max_memory={i: "75GiB" for i in range(4)}
        # device_map="balanced_low_0"
    )
    model.eval()
    print('Model loaded')
    # 打印模型加载后的显存占用情况
    processor = AutoProcessor.from_pretrained(cache_dir)

    # 加载数据集
    with open("./dataset.json", "r") as f:
        dataset_config = json.load(f)
    dataset_path = dataset_config[task]
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # 批量处理数据
    for batch in tqdm(batchify(data, batch_size)):
        # 构造批量 conversation 和图像数据
        conversations = []
        images_list = []
        for item in batch:
            qs = item["question"]
            # images = [Image.open(image).convert('RGB') for image in item['image_path']]

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img, "max_pixels":512*512,} for img in item['image_path']
                    ]
                }
            ]
            conversation[0]['content'].append({
                "type": "text",
                "text": qs + "Let's think step by step. Enclose the reasoning within <think></think> and the answer within <answer></answer>.",
            })
            conversations.append(conversation)

        # 处理文本和图像
        texts = [
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Batch Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_texts)
            # 保存预测结果到对应的批次数据
            for item, pred in zip(batch, output_texts):
                item['pred'] = pred

    # 保存结果
    output_dir = f"/mnt/petrelfs/radshare/reasonbench/VRBench/output/qwenvl72b/"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{task}.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Qwen2VL inference with batch processing.")
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate (e.g., mimic_binary, mimic_multi, etc.)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    args = parser.parse_args()

    main(args.task, args.batch_size)


