import os
import json
import re
import csv
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
# 输入和输出目录
from RaTEScore import RaTEScore
ratescore = RaTEScore()
# model_list = ["ours_cold_final_150", "chexagent3b", "ours_reason_final_150", "ours_process_final_150"]
model_list = [
    # 'qwenvl72b',
    # 'gpt4o',
    # ablation
    'ours_process_final_150',

    # 'qwenvl7b',
    'chexagent3b',
    # 'deepseek',
    # 'llavanext',
    # 'meddr'
]
# model_list = ["ours_rl"]
def query(input_messages) -> str:
    try:
        client = OpenAI(
            # base_url="https://aigptapi.com/v1/",
            base_url="YOUR-BASE-URL-HERE",
            api_key="YOUR-API-KEY-HERE"
        )
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=input_messages,
        )
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in query: {e}")
        return "Error"
    
def get_accuracy(pred, question, answer):
    input_message = [
        {
            "role":"system",
            "content":"You are a medical AI assistant. "
        },
        {
            
            "role":"user",
            "content":f"""
            # Context
            1. Quesiton: {question}
            2. Correct Answer: {answer}
            3. Model Prediction: {pred}
            
            # Goal
            Your task is to evaluate whether a model's prediction is correct. Above, I provide the question, the correct answer, and the model's prediction .

            4. **Output Format**:
            - Return a JSON object with:
                - "correct": a single word "yes" or "no". 
            - Response directly with json object, Do not explain or return irrelevant content.
            """
        }
    ]
    # 第一次尝试
    api_response = query(input_message)
    print(f"API response: {api_response}")
    try:
        api_response = api_response.strip().replace('```json','').replace('```','')
        return json.loads(api_response)
    except json.JSONDecodeError as e:
        print(f"First attempt JSONDecodeError occurred: {e}")
        print("Retrying API call...")
        
        # 第二次尝试
        api_response = query(input_message)
        try:
            api_response = api_response.strip().replace('```json','').replace('```','')
            return json.loads(api_response)
        except json.JSONDecodeError as e:
            print(api_response)
            print(f"Second attempt JSONDecodeError occurred: {e}")
            return {"reason_steps": api_response, "refined_steps": [], "final_conclusion": ""}
    except KeyError as e:
        print(f"API 响应缺少必要字段，跳过样本: {api_response}")
        return {"error_response": api_response, "refined_steps": [], "final_conclusion": ""}

for model in model_list:
    input_dir = f"./output/{model}"
    output_dir = f"./res/{model}"
    os.makedirs(output_dir, exist_ok=True)

    # 需要处理的文件列表
    files_to_process = [
        # "chexpert_binary_test_option",
        # "chexpert_multi_test_option",
        # "chexpert_multi_test_no_option",
        # "chexpert_single_test_option",
        # "mimic_binary_test_option",
        # "mimic_multi_test_no_option",
        # "mimic_multi_test_option",
        # "mimic_single_test_option",
        # "rsna_binary_test_option",
        "siim_binary_test_option",
        # "temporal_cls_test_option",
    ]

    # CSV 文件路径
    csv_path = os.path.join("./res/", "test_results.csv")
    # 检查 CSV 文件是否存在，如果存在则读取现有数据
    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, None)  # 读取表头
            if headers:
                existing_rows = list(reader)
    else:
        # 创建新的 CSV 文件并写入表头
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Model", "Task", "Recall", "Precision", "Efficiency", "Valid Samples Number", "Accuracy", "Samples"])
    total_num =0 
    # 处理每个文件
    for filename in tqdm(files_to_process):
        filename += '.json'
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 从文件名提取任务名称
        task_name = filename.split('.')[0]
        
        # 读取原始数据
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"正在处理文件: {filename} Length: {len(data)}")
        total_num += len(data)
        # continue
        # 准备批量计算所需的数据
        results = []
        correct_count = 0
        total_valid_items = 0
        # breakpoint()
        if 'multi' in filename and 'no_option' in filename:
            # 准备批量计算的列表
            all_preds = []
            all_answers = []
            item_indices = []  # 用于记录有效项的索引
            
            for idx, item in enumerate(data):
                if model in ['chexagent3b','radfm','gpt4o','meddr','deepseek','llavanext']:
                    pred = item.get("pred", "")
                    answer = item.get("answer", "")
                    
                    # 检查 pred 和 answer 是否为空
                    if not pred or not answer:
                        # 创建空结果
                        results.append({
                            "question": item.get("question", ""),
                            "pred": pred,
                            "answer": answer,
                            "score": 0  # 空输入得分为0
                        })
                        continue
                    
                    all_preds.append(pred)
                    all_answers.append(answer)
                    item_indices.append(idx)
                    total_valid_items += 1
                    
                    # 预先创建结果结构（得分稍后填充）
                    results.append({
                        "question": item.get("question", ""),
                        "pred": pred,
                        "answer": answer,
                        "score": None  # 稍后填充
                    })
                elif 'qwen' in model:
                    # breakpoint()
                    pred = item.get("pred", "").split('assistant')[-1]
                    answer = item.get("answer", "")
                    # breakpoint()
                    # 从 pred 中提取 <answer> 标签内容
                    # breakpoint()
                    if '<answer>' in pred and '</answer>' in pred:
                        pred_answer_match = re.search(r"<answer>([\s\S]*?)</answer>", pred)
                        pred_answer = pred_answer_match.group(1).strip() if pred_answer_match else ""
                        pred_answer = " ".join(pred_answer.split())  # 去除多余空格，保留单词间单个空格
                    elif '<answer>' in pred and '</answer>' not in pred:
                        pred_answer = pred.split('<answer>')[-1].strip()
                    elif '</answer>' not in pred and '<answer>' not in pred:
                        pred_answer = pred
                        
                    if '<answer>' in answer:
                        # 从 answer 中提取 <answer> 标签内容
                        gt_answer_match = re.search(r"<answer>(.*?)</answer>", answer)
                        gt_answer = gt_answer_match.group(1).strip() if gt_answer_match else ""
                        gt_answer = " ".join(gt_answer.split())  # 去除多余空格，保留单词间单个空格
                    else:
                        gt_answer = answer
                    # 检查 pred_answer 和 gt_answer 是否为空
                    if not pred_answer or not gt_answer:
                        results.append({
                            "question": item.get("question", ""),
                            "pred": pred,
                            "answer": answer,
                            "pred_answer": pred_answer,
                            "gt_answer": gt_answer,
                            "score": 0  # 空输入得分为0
                        })
                        continue
                    
                    all_preds.append(pred_answer)
                    all_answers.append(gt_answer)
                    item_indices.append(idx)
                    total_valid_items += 1
                    
                    # 预先创建结果结构（得分稍后填充）
                    results.append({
                        "question": item.get("question", ""),
                        "pred": pred,
                        "answer": answer,
                        "pred_answer": pred_answer,
                        "gt_answer": gt_answer,
                        "score": None  # 稍后填充
                    })
                else:
                    pred = item.get("pred", "").split('assistant')[-1]
                    answer = item.get("answer", "")
                    
                    # 从 pred 中提取 <answer> 标签内容
                    pred_answer_match = re.search(r"<answer>(.*?)</answer>", pred)
                    pred_answer = pred_answer_match.group(1).strip() if pred_answer_match else ""
                    pred_answer = " ".join(pred_answer.split())  # 去除多余空格，保留单词间单个空格
                    
                    # 从 answer 中提取 <answer> 标签内容
                    gt_answer_match = re.search(r"<answer>(.*?)</answer>", answer)
                    gt_answer = gt_answer_match.group(1).strip() if gt_answer_match else ""
                    gt_answer = " ".join(gt_answer.split())  # 去除多余空格，保留单词间单个空格
                    
                    # 检查 pred_answer 和 gt_answer 是否为空
                    if not pred_answer or not gt_answer:
                        results.append({
                            "question": item.get("question", ""),
                            "pred": pred,
                            "answer": answer,
                            "pred_answer": pred_answer,
                            "gt_answer": gt_answer,
                            "score": 0  # 空输入得分为0
                        })
                        continue
                    
                    all_preds.append(pred_answer)
                    all_answers.append(gt_answer)
                    item_indices.append(idx)
                    total_valid_items += 1
                    
                    # 预先创建结果结构（得分稍后填充）
                    results.append({
                        "question": item.get("question", ""),
                        "pred": pred,
                        "answer": answer,
                        "pred_answer": pred_answer,
                        "gt_answer": gt_answer,
                        "score": None  # 稍后填充
                    })
            # breakpoint()
            # 批量计算得分
            if all_preds and all_answers:
                assert len(all_preds) == len(all_answers)
                scores = ratescore.compute_score(all_preds, all_answers)
                
                # 将得分填充回结果
                for i, idx in enumerate(item_indices):
                    if i < len(scores):
                        results[idx]["score"] = scores[i]
                
                # 计算平均得分
                total_score = sum(scores)
                accuracy = total_score / total_valid_items if total_valid_items > 0 else 0
            else:
                accuracy = 0
                
        else:  # 二分类任务
            for idx, item in enumerate(data):
                if model == 'chexagent3b':
                    pred = item.get("pred", "")
                    answer = item.get("answer", "")
                    
                    # 检查 pred 和 answer 是否为空
                    if not pred or not answer:
                        is_correct = False
                    else:
                        is_correct = answer in pred
                        total_valid_items += 1
                        if is_correct:
                            correct_count += 1
                    
                    # 存储结果
                    results.append({
                        "question": item.get("question", ""),
                        "pred": pred,
                        "answer": answer,
                        "is_correct": is_correct
                    })
                elif 'ours' in model:
                    pred = item.get("pred", "").split('assistant')[-1]
                    answer = item.get("answer", "")
                    
                    # 从 pred 中提取 <answer> 标签内容
                    pred_answer_match = re.search(r"<answer>(.*?)</answer>", pred)
                    pred_answer = pred_answer_match.group(1).strip() if pred_answer_match else ""
                    pred_answer = " ".join(pred_answer.split())  # 去除多余空格，保留单词间单个空格
                    
                    # 从 answer 中提取 <answer> 标签内容
                    gt_answer_match = re.search(r"<answer>(.*?)</answer>", answer)
                    gt_answer = gt_answer_match.group(1).strip() if gt_answer_match else ""
                    gt_answer = " ".join(gt_answer.split())  # 去除多余空格，保留单词间单个空格
                    
                    # 检查 pred_answer 和 gt_answer 是否为空
                    if not pred_answer or not gt_answer:
                        is_correct = False
                    else:
                        is_correct = pred_answer.lower() in gt_answer.lower()
                        total_valid_items += 1
                        if is_correct:
                            correct_count += 1
                    
                    # 存储结果
                    results.append({
                        "question": item.get("question", ""),
                        "pred": pred,
                        "answer": answer,
                        "pred_answer": pred_answer,
                        "gt_answer": gt_answer,
                        "is_correct": is_correct
                    })
                elif 'qwenvl' in model:
                    pred = item.get("pred", "")
                    answer = item.get("answer", "")

                    if '<answer>' in pred and '</answer>' in pred:
                        # 从 pred 中提取 <answer> 标签内容
                        pred_answer_match = re.search(r"<answer>([\s\S]*?)</answer>", pred)
                        pred_answer = pred_answer_match.group(1).strip() if pred_answer_match else ""
                        pred_answer = " ".join(pred_answer.split())  # 去除多余空格，保留单词间单个空格
                    elif '<answer>' in pred and '</answer>' not in pred:
                        pred_answer = pred.split('<answer>')[-1].strip()
                    else:
                        pred_answer = pred
                    
                    if '<answer>' in answer and '</answer>' in answer:
                        # 从 answer 中提取 <answer> 标签内容
                        gt_answer_match = re.search(r"<answer>(.*?)</answer>", answer)
                        gt_answer = gt_answer_match.group(1).strip() if gt_answer_match else ""
                        gt_answer = " ".join(gt_answer.split())  # 去除多余空格，保留单词间单个空格
                    else:
                        gt_answer = answer
                    
                    # 检查 pred_answer 和 gt_answer 是否为空
                    if not pred_answer or not gt_answer:
                        is_correct = False
                    else:
                        is_correct = gt_answer in pred_answer
                        total_valid_items += 1
                        if is_correct:
                            correct_count += 1
                    
                    # 存储结果
                    results.append({
                        "question": item.get("question", ""),
                        "pred": pred,
                        "answer": answer,
                        "pred_answer": pred_answer,
                        "gt_answer": gt_answer,
                        "is_correct": is_correct
                    })
                else :
                    pred = item.get("pred", "")
                    answer = item.get("answer", "")
                    question = item.get("question", "")

                    response = get_accuracy(pred, question, answer)
                    # response = 'no'
                    if 'yes' in response["correct"]:
                        is_correct = True
                        correct_count += 1
                    else:
                        is_correct = False

                    results.append({
                        "question": question,
                        "pred": pred,
                        "answer": answer,
                        "is_correct": is_correct
                    })
            # breakpoint()
            # 计算准确率
            accuracy = correct_count / total_valid_items if total_valid_items > 0 else 0
        
        # 保存处理后的数据
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # 创建要添加到 CSV 的新行
        new_row = [
            model,                 # Model
            task_name,             # Task
            "0.0",                 # Recall (默认值)
            "0.0",                 # Precision (默认值)
            "0.0",                 # Efficiency (默认值)
            str(total_valid_items), # Valid Samples Number
            f"{accuracy:.16f}",    # Accuracy
            "0.0"                  # Samples (默认值)
        ]
        
        # 追加到现有数据中
        existing_rows.append(new_row)
        
        print(f"处理完成：{filename} -> {output_path}") 
        print(f"有效条目数: {total_valid_items}, 总条目数: {len(data)}")
        if 'multi' not in filename:
            print(f"正确条目数: {correct_count}")
        print(f"平均正确率: {accuracy:.4f}")
        print("-" * 50)
    print(f"总条目数: {total_num} 平均：{total_num/len(files_to_process):.4f}")

    # 将所有数据写入 CSV 文件
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Task", "Recall", "Precision", "Efficiency", "Valid Samples Number", "Accuracy", "Samples"])
        writer.writerows(existing_rows)

    print(f"已将所有结果保存到 {csv_path}")