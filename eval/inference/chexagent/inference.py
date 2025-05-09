import json
import os
from collections import defaultdict
from tqdm import tqdm
from models import CheXagent8B  # 假设使用8B模型
import argparse
def evaluate_on_custom_dataset(model, dataset_path, save_dir):
    # 加载自定义数据集
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # 初始化统计
    results = {
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
        'details': []
    }
    
    # 创建保存目录
    res = []
    # 开始评估
    for sample in tqdm(dataset, desc="Evaluating"):
        try:
            # 准备输入
            img_path = sample['image_path']  # 取第一个图像路径
            question = sample['question']+ " Let's think step by step, then answer the question."
            target = sample['answer']  # 正确答案
            
            # 模型推理
            prompt = question
            # prompt = model.get_prompt(question, options)
            response = model.generate(img_path, prompt, do_sample=False)
            print(response)
            sample['pred'] = response
            res.append(sample)
        except Exception as e:
            print(f"Error processing sample {sample.get('item_id', 'unknown')}: {str(e)}")
            continue
    
    
    # 保存结果
    with open(save_dir, 'w') as f:
        json.dump(res, f, indent=4)
    
    print(f"Results saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run ChexAgent inference on a specific task.")
    parser.add_argument("--task", type=str, required=True, help="Task name to evaluate (e.g., mimic_binary, mimic_multi, etc.)")
    args = parser.parse_args()
    # 配置
    with open("/mnt/petrelfs/radshare/reasonbench/VRBench/inference/dataset.json", "r") as f:
        dataset_config = json.load(f)
    datapath = dataset_config[args.task]
    output_dir = "/mnt/petrelfs/radshare/reasonbench/VRBench/output/chexagent8b/"
    os.makedirs(output_dir, exist_ok=True)  # 创建目录，如果已存在则忽略
    outputpath = os.path.join(output_dir, f"{args.task}.json")
    
    # 加载模型
    model = CheXagent8B()
    
    # 开始评估
    evaluate_on_custom_dataset(model, datapath, outputpath)

if __name__ == '__main__':
    main()