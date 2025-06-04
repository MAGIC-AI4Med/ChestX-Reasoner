import pandas as pd
from PIL import Image
import io
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    
    train_dataset=datasets.load_dataset('json', data_files=["/mnt/lustre/fanziqing/radfm/radshare/RadFM_final/verl/data/processed/cold/merge_train.json"])
    test_dataset=datasets.load_dataset('json', data_files=["/mnt/lustre/fanziqing/radfm/radshare/RadFM_final/verl/data/processed/cold/merge_test.json"])

    instruction_following = r"Let's think step by step. Enclose the reasoning within <think></think> and the answer within <answer></answer>."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('question')
            answer = example.pop('answer')
            images = example.pop('image_path')
          
            for item in images:
                problem="<image>"+problem
            
            prompt = problem + "\n"+instruction_following

            data_source=example.pop('data_source')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "images": images,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)


    train_dataset["train"].to_parquet(os.path.join("./", 'train_rl.parquet'))
    test_dataset["train"].to_parquet(os.path.join("./", 'test_rl.parquet'))
