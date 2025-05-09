import pandas as pd
import numpy as np
import openai
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import random

report_data_path = '/path/to/your/csv'  
file_name = "mimic_multi_v5_3"
start_index = 3000
end_index = 5000


# 疾病列表
disease_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion','Pneumonia', 'Pneumothorax','Support Devices']




findings_file_path = "./findings.json"
num_paralled = 8

def query(input_messages) -> str:
    try:
        client = OpenAI(
            # base_url="https://aigptapi.com/v1/",
            # api_key="sk-iN6LzlZNcIWhlJu9rvF3Cq5bllSROwkege5lqeHD625c8smA"
            base_url="https://api.gpts.vin/v1",
            # api_key="sk-Ht9UI5FprlzkYn8WvTt2z4H8Bq5rTK63LeUomdYKB9DYHyku"
            api_key="sk-9gaS4kbirRrLTSsbzEX2GDQYJ2PUco74WXmcOoi8I9BI9YrZ"
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=input_messages,
        )
        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in query: {e}")
        return ""


def generate_qa_and_analysis_plan(vqa_instance, data_item):
    """
    第一次调用API
    输入原报告
    生成问题，答案和计划
    """
    origin_report = data_item['original_report']
    with open(findings_file_path, 'r') as f:
        findings = json.load(f)
    system_message = {
        "role": "system",
        "content": "You are a medical AI assistant that generates diagnostic analysis plans based on clinical reports."
    }
    
    user_message = {
        "role": "user",
        "content": f"""
        ## Context:
        1. Observed Image feature and information: {origin_report}
        2. Question: {vqa_instance['q']}
        3. Answer: {vqa_instance['a']}
        4. Options: {vqa_instance['options']}

        ## Goal:
        1. Generate a corresponding reason plan for disease types based on medical knowledge and logic, such as what kind of manifestations the symptom or disease might have, and based on which findings the disease can be determined. The reason plan should include all diseases in each options. (support devices mean medical device like catheter) 
        2. The reasoning plan should be based on medical knowledge and logic, e.g., what are the possible manifestations of each symptom or disease, arranging for the analysis and interpretation of each option. Note: An option can only be selected if all the findings in that option are satisfied, if partially correct it should be excluded. The reason plan should include analysis steps for all findings in all options.

        ## Rule:
        1. The generated plan should only include observation and reason plan of the images. Performing other clinical tests, clinical history should not be considered in the analysis plan. 
        2. At the beginning of your plan, you should analyze the problem and options, then list the medical knowledge to plan which areas to judge and why. Use wording such as: the problem requires analysis of findingA, B, C, which xxx, I should examine xxx.
        Response in following json format:
        {{
            "plan":{{
                "disease name":"plan",
                "disease name":"plan",
                "disease name":"plan",
                "disease name":"plan",
                ...
            }}
        }}
        """
    }

    api_response = query([system_message, user_message])
    
    try:
        api_response = api_response.strip().replace('```json','').replace('```','')
        qa_analysis = json.loads(api_response)
        return qa_analysis
    except json.JSONDecodeError as e:
        print(f"First Step JSONDecodeError occurred: {e}")
        return {
            "error_response": api_response
        }
    except KeyError as e:
        print(f"API 响应缺少必要字段，跳过样本: {api_response}")
        return {"error_response": api_response, "reason_steps": [], "q": "", "a": "", "options": ""}

def generate_analysis_from_plan(data_item, analysis_plan):
    """
    第二次调用API
    输入第一步生成的计划，
    生成对应的推理步骤
    """
    origin_report = data_item['original_report']
    
    system_message = {
        "role": "system",
        "content": "You are a medical AI assistant that generates detailed analysis based on the given analysis plan and original clinical report."
    }
    
    user_message = {
        "role": "user",
        "content": f"""
        ## Context:
        1. Analysis Plan: {analysis_plan}
        2. Observed Image feature and information: {origin_report}

        ## Goal:
        Generate structured reason steps for each disease based on the analysis plan and given image information.

        ## Rule:
        1. If the object to be analyzed in the analysis plan is not mentioned in the image observation information given, then the indications of the object to be analyzed are considered normal.
        2. Each reasoning step should have strucutres like, From given images, we observed xx, diagnosis of this disease. Analyze one disease in each reasoning steps.
        3. Return in following json format:
        {{
            "disease name":"reason_step",
            ...
        }}
        """
    }

    api_response = query([system_message, user_message])
    
    try:
        api_response = api_response.strip().replace('```json','').replace('```','')
        return json.loads(api_response)
    except json.JSONDecodeError as e:
        print(f"Second Step JSONDecodeError occurred: {e}")
        return {
            "reason_steps":api_response,
            "error_response": api_response,
            "analysis_steps": [],
            "conclusions": "",
            "metadata": data_item.get("original_report", "")
        }
    except KeyError as e:
        print(f"API 响应缺少必要字段，跳过样本: {api_response}")
        return {"error_response": api_response, "analysis_steps": [], "conclusions": ""}

def refine_reasoning_steps(vqa_instance, analysis_content):
    """
    第三次调用API
    输入问题，计划，推理步骤和回答
    润色推理步骤，增加逻辑性
    """
    reasoning_steps = analysis_content
    
    system_message = {
        "role": "system",
        "content": "You are a medical AI assistant that refines reasoning steps to ensure medical logic consistency."
    }
    
    user_message = {
        "role": "user",
        "content": f"""
        ## Context:
        Question: {vqa_instance['q']}
        Diagnosis Plan: {vqa_instance['plan']}
        Original Reasoning Steps: {reasoning_steps}
        Answer:{vqa_instance['a']}

        ## Goal:
        Refine the reasoning steps to ensure they are logically consistent and clear.

        ## Rule:
        1. There may be words like "according to the original report", "the report mentioned" or "the findings note" in the reasoning steps, please remove these words and rectify the language. The orginal report should not be mentioned in the reasoning steps.
        2. There may be words like "according to the analysis plan" and "with the analysis plan" in the reasoning steps, please remove and polish the language. The analysis plan should not be mentioned in the reasoning steps.
        2. Modify the reasoning process into a complete analytical observation and judgment process: add words such as transitions to make the logic flow more smoothly.
        3. Keep all analysis about all diseases in original reasoning steps in the generated reason_steps.
        4. Add a step of conclusion to get the final answer at the end of the reasoning process, like: "In conclusion, the final answer is ...".
        5. Return in following json format: {{
            "reason_steps":[
                "",
                "",
                ...
                ""
            ]
        }}
        """
    }

    api_response = query([system_message, user_message])
    
    try:
        api_response = api_response.strip().replace('```json','').replace('```','')
        return json.loads(api_response)
    except json.JSONDecodeError as e:
        print(f"Third Step JSONDecodeError occurred: {e}")
        return {"reason_steps": api_response, "refined_steps": [], "final_conclusion": ""}
    except KeyError as e:
        print(f"API 响应缺少必要字段，跳过样本: {api_response}")
        return {"error_response": api_response, "refined_steps": [], "final_conclusion": ""}

def generate_vqa_instance(data_item):
    try:
        # breakpoint()
        options = data_item['findings']
        # 生成问题和答案
        all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
        pos_label = [x for x in options if options[x] == 1 and x in all_labels]
        if len(pos_label) < 1 or len(pos_label) > 5:
            return None
        # construct negative options
        neg_findings = []
        for i in range(3):
            neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
            while neg_sample == set(pos_label):
                neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
            neg_findings.append(neg_sample)
        pos_findings = ", ".join(pos_label).lower()
        neg_findings = [", ".join(x).lower() for x in neg_findings]
        # construct options
        options = {pos_findings: 1}
        options.update({item: 0 for item in neg_findings})
        # shuffle the option
        items = list(options.items())
        random.shuffle(items)
        options = dict(items)
        # add option style
        options_a = [k for k, v in options.items() if v]
        options = [k for k, v in options.items()]
        options_str = "\n".join(options)
        question_str = f"Which findings are in this chest X-ray? Options{options_str}"
        vqa_instance = {
            "q": "Which findings are in this chest X-ray?",
            "options": options,
            "a": pos_findings
        }
        # 生成推理过程
        plan = generate_qa_and_analysis_plan(vqa_instance, data_item)
        # print('b')
        vqa_instance.update({
            "plan":plan['plan']
        })
        instance_2 = generate_analysis_from_plan(data_item, plan['plan'])
        # print('c')
        reason_steps = instance_2
        instance_3 = refine_reasoning_steps(vqa_instance, reason_steps)
        # print('d')
        vqa_instance.update({
            "reason_steps": instance_3,
            "label": data_item['label'],
            "value": data_item['value'],
            "images_path": data_item['images_path'],
            "original_reason_steps": reason_steps,
            "original_report": data_item['original_report'],
            "findings": data_item['findings']
        })
        return vqa_instance
    except Exception as e:
        print(f"generate error : {e}")
        return {}

def generate_vqa_data_parallel(report_data):
    """Parallelize the process of generating VQA instances."""
    vqa_data = []
    with ThreadPoolExecutor(max_workers=num_paralled) as executor:
        results = list(tqdm(executor.map(
            lambda data_item: generate_vqa_instance(data_item),
            report_data
        ), total=len(report_data), desc="Generating VQA Instances"))
        vqa_data.extend(results)
    return vqa_data

def translate_to_chinese(text):
    try:
        system_message = {
            "role": "system",
            "content": "You are an AI assistant specialized in language translation. Translate the given text into Chinese."
        }
        user_message = {
            "role": "user",
            "content": f"Translate the following text into Chinese: {text}"
        }

        api_response = query([system_message, user_message])
        return api_response.strip()
    except Exception as e:
        print(f"Error during translation: {e}")
        return ""

def translate_column_parallel(column_data):
    """Parallel translate for a single column."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        translated_data = list(tqdm(
            executor.map(translate_to_chinese, column_data),
            total=len(column_data),
            desc="Translating column"
        ))
    return translated_data

def translate_all_columns_parallel(df):
    """Parallel translate all columns in the dataframe."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(
                translate_column_parallel,
                [df[col].tolist() for col in df.columns]
            ),
            total=len(df.columns),
            desc="Translating all columns"
        ))
    
    for i, col in enumerate(df.columns):
        df[col] = results[i]
    return df

# 读取CSV数据
def convert(disease_df):
    report_data = []
    for _, row in disease_df.iterrows():
        labels_values = {
            key: (value if pd.notna(value) and value in [1, -1, 0] else 0)
            for key, value in row.items()
            if key in disease_columns
        }
        report_item = {
            'original_report': f"""FINDINGS: {row['findings']}
            IMPRESSION: {row['impression']}""",
            'images_path': eval(row['images_path']) if isinstance(row['images_path'], str) else row['images_path'],
            'findings': labels_values,
            'label':'',
            'value':''
        }
        report_data.append(report_item)
        # breakpoint()
    return report_data

def main():
    # 读取CSV数据
    df = pd.read_csv(report_data_path)
    vqa_data = []
    df['label_count'] = df[disease_columns].apply(lambda row: (row == 1).sum(), axis=1)
    disease_df = df[df['label_count'] > 2]
    report_data = convert(disease_df)
    print('begin')
    vqa_data = generate_vqa_data_parallel(report_data[start_index:end_index])

    # 保存JSON格式
    os.makedirs('./report_cold_start_vqa', exist_ok=True)
    with open(f'./report_cold_start_vqa/{file_name}.json', 'w', encoding='utf-8') as f:
        json.dump(vqa_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

