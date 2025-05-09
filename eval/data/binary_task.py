import pandas as pd
import openai
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json


# You need to prepare the annotation csv of mimic-iv, and add it below.
report_data_path = '/path/to/your/csv'  
file_name="mimic_binary"
print(f"Souce File:{report_data_path}")
print(f'save to: ./report_cold_start_vqa/{file_name}.json')



disease_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 
# 疾病列表,没有 no finding, pleural other
                   'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Support Devices', 
                   'Pneumonia', 'Pneumothorax']
task_name = "Binary Disease Classification"
# start_index = 2
# end_index = 3
sample_per_disease = 200
end_index=20
start_index=0


def query(input_messages) -> str:
    try:
        client = OpenAI(
            # base_url="https://aigptapi.com/v1/",
            base_url="your-api-base-url",
            api_key="your-api-key"
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


def generate_qa_and_analysis_plan(data_item):
    """
    第一次调用API
    输入原报告
    生成问题，答案和计划
    """
    origin_report = data_item['original_report']
    
    system_message = {
        "role": "system",
        "content": "You are a medical AI assistant that generates diagnostic Q&A and analysis plans based on clinical reports."
    }
    
    user_message = {
        "role": "user",
        "content": f"""
        ## Context:
        1. Observed Image feature and information: {origin_report}
        2. Question: {data_item['q']}
        3. Answer: {data_item['a']}

        ## Goal:
        1. Generate a corresponding reason plan for disease in the question based on medical knowledge and logic, such as what kind of manifestations the symptom or disease might have, and based on which findings the disease can be determined. The reason plan should include diseases for {data_item['label']}. (support devices mean medical device like catheter)

        ## Rule:
        1. The generated plan should only include observation and reason plan of the images. Performing other clinical tests, clinical history should not be considered in the analysis plan. 
        2. At the beginning of your plan, you should analyze the problem and list the medical knowledge to plan which areas to judge and why. Use wording such as: the problem requires analysis of a certain disease, I should examine xxx.
        3. Response in following json format :
        {{
            "plan":{{
                "disease":"plan",
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
        Generate detailed reason steps based on the analysis plan and given image information.

        ## Rule:
        1. If the object to be analyzed in the analysis plan is not mentioned in the image observation information given, then the indications of the object to be analyzed are considered normal.
        2. Each reasoning step should have strucutres like, From given images, we observed xx, diagnosis of this disease. Analyze one disease in reasoning step.
        3. Return in following json format:
        {{
            "reason_step":" ..."
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
        Reasoning Steps: {reasoning_steps}
        Answer:{vqa_instance['a']}

        ## Goal:
        Refine the reasoning steps to ensure they are logically consistent and clear.

        ## Rule:
        1. There may be words like "according to the original report" and "the report mentioned" in the reasoning steps, please remove them and rectify the language.
        2. There may be words like "according to the analysis plan" and "with the analysis plan" in the reasoning steps, please remove and polish the language. The analysis plan should not be mentioned in the reasoning steps.
        2. Modify the reasoning process into a complete analytical observation and judgment process: add words such as transitions to make the logic flow more smoothly.
        3. The reasoning process should begin with an analysis of the problem and planning of the program, and end with arriving at the answer. The conclusion of the reasoning should align with the anwer: {vqa_instance['a']}.
        4. If a step in the reasoning process is not directly related to determining the answer to a question, is of secondary importance, or does not affect the answer choice, delete it.
        5. Return in following json format: {{
            "reason_step":" ... "
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
        # print('a')
        vqa_instance = {
            "q":f"Does this chest X-ray show {data_item['label'].lower()}?",
            "options":[
                "Yes",
                "No"
            ],
            "a":'Yes' if data_item['value'] == 1 else 'No',
            "original_report":data_item['original_report'],
            "label":data_item['label']
        }
        plan = generate_qa_and_analysis_plan(vqa_instance)
        vqa_instance.update({
            'plan':plan
        })
        # print('b')
        instance_2 = generate_analysis_from_plan(data_item, vqa_instance['plan'])
        # print('c')
        reason_steps = instance_2['reason_step']
        instance_3 = refine_reasoning_steps(vqa_instance, reason_steps)
        # print('d')
        vqa_instance.update({
            "reason_steps": instance_3['reason_step'],
            "label": data_item['label'],
            "value": data_item['value'],
            "images_path": data_item['images_path'],
            "original_reason_steps": reason_steps,
            "original_report": data_item['original_report']
        })
        return vqa_instance
    except Exception as e:
        print(f"generate error : {e}")
        return {}

def generate_vqa_data_parallel(report_data):
    """Parallelize the process of generating VQA instances."""
    vqa_data = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(
            lambda data_item: generate_vqa_instance(data_item),
            report_data
        ), total=len(report_data), desc="Generating VQA Instances"))
        vqa_data.extend(results)
    return vqa_data

# 读取CSV数据
def convert(disease_df, disease):
    report_data = []
    for _, row in disease_df.iterrows():
        report_item = {
            'original_report': f"""FINDINGS: {row['findings']}
            IMPRESSION: {row['impression']}""",
            'images_path': eval(row['images_path']) if isinstance(row['images_path'], str) else row['images_path'],
            'label':disease,
            'value':row[disease]
        }
        report_data.append(report_item)
    return report_data

def main():
    # 读取CSV数据
    df = pd.read_csv(report_data_path)
    # breakpoint()
    if "mimic" in file_name:
        df = df[df['split'] == 'train']
    vqa_data = []
    for disease in disease_columns[start_index:end_index]:
        # 获取正样本
        positive_df = df[df[disease] == 1]
        # 获取负样本
        negative_df = df[(df[disease] == 0) | (df[disease].isna())]
                
        # 如果正样本不足sample_per_disease条，全部取用
        if len(positive_df) < sample_per_disease:
            sampled_positive = positive_df
        else:
            sampled_positive = positive_df.sample(n=sample_per_disease, random_state=50)
            
        # 如果负样本不足sample_per_disease条，全部取用
        if len(negative_df) < sample_per_disease:
            sampled_negative = negative_df
        else:
            sampled_negative = negative_df.sample(n=sample_per_disease, random_state=50)
            
        # 合并正负样本
        disease_df = pd.concat([sampled_positive, sampled_negative])
        print(f"len disease df:{len(disease_df)}")
        print(f"Processing {disease}: {len(sampled_positive)} positive, {len(sampled_negative)} negative samples")
        report_data = convert(disease_df, disease)
        print(f"len report data: {len(report_data)}")
        vqa_data_item = generate_vqa_data_parallel(report_data)
        vqa_data.extend(vqa_data_item)

        with open(f'./report_cold_start_vqa/{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(vqa_data, f, indent=4, ensure_ascii=False)

    # 保存JSON格式
    os.makedirs('./report_cold_start_vqa', exist_ok=True)
    with open(f'./report_cold_start_vqa/{file_name}.json', 'w', encoding='utf-8') as f:
        json.dump(vqa_data, f, indent=4, ensure_ascii=False)

    # 保存CSV格式
    try:
        if vqa_data:
            df = pd.DataFrame.from_records(vqa_data)
            df.to_csv(
                f'./report_cold_start_vqa/{file_name}.csv',
                index=False,
                encoding='utf-8'
            )
    except Exception as e:
        print(f"Error while saving to CSV: {e}")

    # # 翻译
    # df = pd.read_csv(f'./report_cold_start_vqa/{file_name}.csv', encoding='utf-8')
    # df = translate_all_columns_parallel(df)
    # df.to_csv(f'./report_cold_start_vqa/{file_name}_translated.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()

