
import pandas as pd
import openai
import argparse
import os
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import random
# 设置代理
num_paralled = 8
version = "v6"
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
    
def get_key_finding(reason_steps):
    input_message = [
        {
            "role":"system",
            "content":"You are a medical AI assistant. "
        },
        {
            
            "role":"user",
            "content":f"""
            # Context
            1. Reason Steps: {reason_steps}

            # Goal
            1. Read the provided Reason Steps carefully.
            2. Identify each key medical observation (e.g., widening of the mediastinum, size of the cardiac silhouette) and determine its status based on the reasoning (e.g., confirmed as xxx, not present, unlikely).
            
            3. Ensure the observations are concise and directly tied to the reasoning steps.
            4. If there is any ambiguity or contradiction in the reasoning, reflect the findings as per the steps provided, and do not infer beyond the given text.
            5. If the reason steps lacks some key findings, 
            5. **Output Format**:
            - Return the extracted key reasoning steps as a JSON list of strings, where each string represents a specific observation or finding from the report.
            - Structure your response in JSON format with the following schema:
            {{
                "key_conclusion": [
                    {{
                        "observation": "<description of the finding>",
                        "status": "<status based on reasoning>"
                    }}
                ]
            }}
            - Respond directly with the JSON object, without explanations or additional content.
            """
        }
    ]
    # 第一次尝试
    api_response = query(input_message)
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
    
def cal_recall(reason_steps, pred):
    input_message = [
        {
            "role":"system",
            "content":"You are a medical AI assistant. "
        },
        {
            
            "role":"user",
            "content":f"""
            # Context
            You need to calculate the recall of key medical findings between ground truth and model predictions. Given two dictionaries:
            1. Ground Truth: {reason_steps}
            2. Model Output: {pred}
            
            
            # Goal
            Your job is to determine how many observations from Ground Truth are recalled in Model Output. Follow these steps:

            1. Compare each observation in Ground Truth with the observations in Model Output.
            2. An observation is considered "recalled" if its meaning or content is substantially present in Model Output, even if the phrasing differs slightly. Focus on semantic equivalence rather than exact wording.
            3. Compile a list of recalled observations from Ground Truth that match Model Output.
            4. Count the total number of recalled observations.

            Output Format:
            - Return a JSON object with:
            - "recalled_findings": A list of dictionaries, each containing the "observation" and "status" from `gt_key_finding` that were recalled.
            - "recalled_count": An integer representing the number of recalled observations.
            - Respond directly with the JSON object, without explanations or additional content.
            """
        }
    ]
    # 第一次尝试
    api_response = query(input_message)
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


    
def get_precision(report, reason_steps):
    input_message = [
        {
            "role":"system",
            "content":"You are a medical AI assistant. "
        },
        {
            
            "role":"user",
            "content":f"""
            # Context
            1. Ground Truth Report: {report}
            2. Reasoning Steps: {reason_steps}
            # Goal
            Your task is to determine which steps in a provided list of reasoning steps are correct based on a chest X-ray report. Follow these instructions:

            1. **Input Data**:
            - "Ground Truth Report": A text string containing the factual observations and findings from a chest X-ray report.
            - "Reasoning Steps": A JSON list of strings, where each string represents a reasoning step (e.g., an observation, finding, or analytical statement) derived from a model output.

            2. **Correctness Criteria**:
            - A reasoning step is **correct** if:
                - It explicitly aligns with or does not contradict the observations/findings in the ground truth report.
                - It analyzes an observation or finding not mentioned in the report, and the report implies it is normal (e.g., absence of mention means no abnormality), and the step concludes it is normal.
            - A reasoning step is **incorrect** if:
                - It contradicts an explicit observation or finding in the ground truth report.
                - It analyzes an observation or finding not mentioned in the report, the report implies it is normal, and the step concludes it is abnormal.
            - General statements (e.g., definitions or procedural intent) are considered correct unless they misrepresent the report.
            3. **Task**:
            - Compare each reasoning step to the ground truth report.
            - Assess its correctness based on the criteria above.
            - Count the number of correct steps.

            4. **Output Format**:
            - Return a JSON object with:
                - "correct_steps": A list of the reasoning steps that are factually correct.
                - "correct_count": An integer representing the number of correct steps.
            - Response directly with json object, Do not explain or return irrelevant content.
            """
        }
    ]
    # 第一次尝试
    api_response = query(input_message)
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

def get_efficiency(reason_steps, pred):
    input_message = [
        {
            "role":"system",
            "content":"You are a medical AI assistant. "
        },
        {
            
            "role":"user",
            "content":f"""
            # Context
            1. Reasoning Steps: {reason_steps}
            2. Model Prediction: {pred}
            
            # Goal
            You are an AI assistant tasked with evaluating the reasoning efficiency of a model's prediction based on a ground truth. I will provide you with two pieces of information:
            1. "Reasoning Steps": A list of 10 reasoning steps representing the ground truth analysis of a chest X-ray.
            2. "Model Prediction": The model's output, which contains its own reasoning and conclusions about the same chest X-ray.

            # Task
            Your task is to:
                - Identify the distinct reasoning steps in "pred".
                - Compare each reasoning step in "pred" to the "reason_steps" to determine if it is supported, contradicted, or unsupported by the ground truth.
                - Calculate the efficiency as the ratio of supported reasoning steps in "pred" to the total number of reasoning steps in "pred" (i.e., supported steps / total steps in pred).
                - Provide a step-by-step explanation of your analysis and the final efficiency score as a percentage.

            # Output Format:
            - Return a JSON object containing:
                - "effective_steps": A list of the reasoning steps from "pred" that are supported by "reason_steps" (include the text of each step).
                - "effective_count": The number of effective steps.
                - "ineffective_count": The number of ineffective steps.
            - Response directly with json object, Do not explain or return irrelevant content.
            """
        }
    ]
    # 第一次尝试
    api_response = query(input_message)
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
            2. Answer: {answer}
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
    

def process_item(item, model):
    try:
        # breakpoint()
        if 'ours' in model:
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
            # breakpoint()
            # 比较结果
            is_correct = pred_answer.lower() in gt_answer.lower()
            item['correct'] = 'yes' if is_correct == True else 'no'
        else:
            item['correct']=get_accuracy(item['pred'],item['question'],item['answer'])['correct']
        item['gt_key_finding'] = get_key_finding(item['reason_steps'])
        item['pred_key_finding'] = get_key_finding(item['pred'])
        item['recall_res'] = cal_recall(item['gt_key_finding'], item['pred_key_finding'])
        # print('Here')
        # breakpoint()
        item['precision_res'] = get_precision(item['original_report'], item['pred_key_finding'])
        item['efficiency'] = get_efficiency(item['reason_steps'], item['pred'])
        return item
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name to evaluate')
    parser.add_argument('--task', type=str, required=True, help='Task name to evaluate')
    parser.add_argument('--id', type=str, required=False, default='0', help='Task name to evaluate')
    args = parser.parse_args()

    # 初始化结果存储
    results = {
        'Model': args.model,
        'Task': args.task,
        'Recall': 0,
        'Precision': 0,
        'Efficiency': 0,
        'Samples': 0
    }
    print(f"model : {args.model}, task:{args.task}")
    # 加载数据
    with open(f"/mnt/petrelfs/radshare/reasonbench/VRBench/output/{args.model}/{args.task}.json","r") as f:
        data = json.load(f)
    data = data
    total_recall = 0
    total_precision = 0
    total_efficiency = 0
    valid_samples = 0
    correct_samples = 0

    evaluation_res = []
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=num_paralled) as executor:
        futures = [executor.submit(process_item, item, args.model) for item in data]
        for future in tqdm(futures, total=len(data)):
            try:
                item = future.result()
                if item is not None:
                    evaluation_res.append(item)
                    # 累加指标
                    # print(f"item recall : {item['recall_res']}")
                                        # 计算 recall
                    gt_conclusion_len = len(item['gt_key_finding']['key_conclusion'])
                    if gt_conclusion_len > 0:
                        total_recall += item['recall_res']['recalled_count'] / gt_conclusion_len
                    
                    # 计算 precision
                    pred_conclusion_len = len(item['pred_key_finding']['key_conclusion'])
                    if pred_conclusion_len > 0:
                        total_precision += item['precision_res']['correct_count'] / pred_conclusion_len
                    pred_total_len = item['efficiency']['effective_count'] + item['efficiency']['ineffective_count']
                    if pred_total_len > 0:
                        total_efficiency += item['efficiency']['effective_count'] / pred_total_len
                    valid_samples += 1
                    if "yes" in item['correct'].lower():
                        correct_samples += 1
            except Exception as e:
                print(f"Error in future result: {e}")
                continue

    # 计算平均指标
    if valid_samples > 0:
        results['Recall'] = total_recall / valid_samples
        results['Precision'] = total_precision / valid_samples
        results['Efficiency'] = total_efficiency / valid_samples
        results['Valid Samples Number'] = valid_samples
        results["Accuracy"] = correct_samples / valid_samples

    save_dir = f"./res/reasoning_res/{args.model}"
    os.makedirs(save_dir, exist_ok=True)
    # 构建保存路径
    save_path = f"{save_dir}/{args.task}_evaluation_res.json"
    with open(save_path, 'w') as f:
        json.dump(evaluation_res, f, indent=2)

    # 保存结果到CSV
    output_file = f"./res/reasoning_res/results.csv"
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=['Model', 'Task', 'Recall', 'Precision', 'Efficiency', 'Valid Samples Number', 'Accuracy'])

    # 更新或添加新结果
    mask = (df['Model'] == args.model) & (df['Task'] == args.task)
    if mask.any():
        df.loc[mask, ['Recall', 'Precision', 'Efficiency', 'Valid Samples Number', 'Accuracy']] = [
            results['Recall'], results['Precision'], results['Efficiency'], results['Valid Samples Number'], results['Accuracy']
        ]
    else:
        # 使用新的_append方法
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
