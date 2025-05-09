import pandas as pd
import openai
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json



template_file_path = './template.json'
findings_file_path = "./findings.json"
temporal_data_path = "/path/to/your/json"
file_name = "temporal_task"
end_index = 2000
with open(temporal_data_path,"r") as f:
    temporal_data = json.load(f)
temporal_data = temporal_data[0]['train']
num_paralled = 8


def query(input_messages) -> str:
    try:
        client = OpenAI(
            api_key="your-api-key",
            base_url="your-api-base-url",
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
    report_1 = data_item['report1']
    report_2 = data_item['report2']

    system_message = {
        "role": "system",
        "content": "You are an imaging specialist who is responsible for analyzing a patient's x-ray image report."
    }
    
    user_message = {
        "role": "user",
        "content": f"""
        ## Context:
        You are given: 
        Two x-ray image reports representing the same patient's image findings at different times:
            Previous report: {report_1}
            Current report: {report_2}

        A qa pair that asks about the progression of a particular symptom, with the answer indicating whether the condition is stabilizing, worsening, or improving.
            Question: {data_item['qa_pair'][0]['q']}
            Answer: {data_item['qa_pair'][0]['a']}

        ## Goal:
        You need to compare the contents of the two reports to generate the reasoning process for this question-answer pair.  

        ## Rule:
        1. Please describe your reasoning process in detail, try to analyze it from the perspective of an radiology physician, don't be seen to be referring to the reports, you need to act as if you have come to your own conclusions by comparing the two images.
        2. The generated reason steps should only include observation and reason content of the images. Performing other clinical tests, clinical history should not be considered in the reason steps. 
        2. At the beginning of your reason steps, you should analyze the problem and list the medical knowledge to plan which areas to compare to get the conclusion. Use wording such as: the problem requires comparison of {{disease}}, I should check xxx.
        3. The answer to this question should be able to be found through image information.
        Response in following json format, generated questions can be more flexible :
        {{
            "reason_step":" ..."
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
    
    system_message = {
        "role": "system",
        "content": "You are a medical AI assistant that generates detailed analysis based on the given analysis plan and original clinical report."
    }
    report_1 = data_item['report1']
    report_2 = data_item['report2']
    user_message = {
        "role": "user",
        "content": f"""
        ## Context:
        1. Analysis Plan: {analysis_plan}
        2. Two x-ray image reports representing the same patient's image findings at different times:
            Previous report: {report_1}
            Current report: {report_2}

        ## Goal:
        Generate detailed reason step based on the analysis plan and given image information.

        ## Rule:
        1. If the object to be analyzed in the analysis plan is not mentioned in the image observation information given, then the indications of the object to be analyzed are considered normal.
        2. Reasoning step should have strucutres like, From given images, we observed xx, diagnosis of the change. Return in following json format:
        {{
            "reason_step":""
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

def refine_reasoning_steps(data_item, analysis_content):
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
        You are given a question-answer pair, and related reasoning steps.
        Question: {data_item['qa_pair'][0]['q']}
        Answer: {data_item['qa_pair'][0]['a']}
        Reasoning Steps: {reasoning_steps}

        ## Goal:
        Refine the reasoning steps to ensure they are logically consistent and clear.

        ## Rule:
        1. There may be words like "according to the original report" and "the report mentioned" in the reasoning steps, please remove them and rectify the language.
        2. There may be words like "according to the analysis plan" and "with the analysis plan" in the reasoning steps, please remove and polish the language. The analysis plan should not be mentioned in the reasoning steps.
        3. Modify the reasoning process into a complete analytical observation and judgment process: add words such as transitions to make the logic flow more smoothly.
        4. The reasoning process should begin with an analysis of the problem and planning of the program, and end with arriving at the answer.
        5. Do not directly locate the disease to a specific area of specific organ at the beginning of the analysis, but locate it to a specific location after observation
        6. Return in following json format: {{
            "reason_steps":""
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
        vqa_instance =  {
            "q": data_item['qa_pair'][1]['q'],
            "options":data_item['options'],
            "a": data_item['qa_pair'][1]['a'],
        }  
        reason_steps = generate_qa_and_analysis_plan(data_item)
        # print(f'first : {reason_steps}')
        refined_reason_steps = refine_reasoning_steps(data_item,reason_steps)
        
        vqa_instance.update({
            "reason_steps":refined_reason_steps,
            "original_reason_steps":reason_steps,
            "original_report_1":data_item['report1'],
            "images_path": data_item['image_path'],
            "original_report_2":data_item['report2']
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
    vqa_data = generate_vqa_data_parallel(temporal_data[:end_index])


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

if __name__ == "__main__":
    main()

