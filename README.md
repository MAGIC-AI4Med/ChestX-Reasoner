<p align="center" width="100%">
</p>

<div id="top" align="center">

ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification.
-----------------------------
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4> |<a href="https://arxiv.org/pdf/2504.20930?"> 📑 Paper </a> |
<a href="https://github.com/MAGIC-AI4Med/ChestX-Reasoner"> 🐱 Github Repo </a> |
<a href="https://huggingface.co/byrLLCC/ChestX-Reasoner"> 🐱 ChestX-Reasoner-7B </a> |
  <a href="https://huggingface.co/byrLLCC/ChestX-Reasoner"> 🐱 RadRBench </a> |
</h4>

<!-- **Authors:** -->
_**Ziqing Fan<sup>1,2 </sup>, Cheng Liang<sup>1,2 </sup>, Chaoyi Wu<sup>1,2 </sup>, Ya Zhang<sup>1,2</sup>, Yanfeng Wang<sup>1,2</sup>, Weidi Xie<sup>1,2</sup>**_


<!-- **Affiliations:** -->

_<sup>1</sup> Shanghai Jiao Tong University,
<sup>2</sup> Shanghai AI Laboratory._

</div>

The official codes for "ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification".  

## Training
In the following, we provide an overview and detailed guidance on the code used to train our ChestX-Reasoner and its variants.  
* Note that SFT step requires at least 4 A100 80GB GPUs and training for about 2 days.  
* Note that RL step requires at least 8 A100 80GB GPUs and training for about 3 days.  
### Environment  
You can install the code environment used for training our model. Our code is established based on **VERL(https://github.com/volcengine/verl)** engine. You may see for more detailed instructions. Besides, we provide a copy of our env list in **./env.txt**.  
```bash
conda create -n env_name python==3.10
conda activate env_name
pip3 install torch torchvision
pip3 install flash-attn --no-build-isolation
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .[vllm]
```
* Python: Version >= 3.9  
* CUDA: Version >= 12.1  
* VLLM: Version >= 0.7  
### Supervised Fine-Tuning  
```bash
cd ChestXReasoner
bash run_SFT.sh
```
Notably, before run the bash file, there are configs and data paths should set in your devices. Please see details in ./ChestXReasoner/readme.md  


### Reinforcement Learning  
To be continue  

### Reinforcement Learning with Process Reward  

## Evaluation

### Benchmark Data  
In `eval/data`, we present our benchmark construction code and our data.
### Evaluation  
We provide:
1. The evaluation code on both reasoning and accuracy in `eval/`
2. The baseline inference code in `eval/inference`
3. The evaluation results on both reasoning and accuracy of all baselines in `eval/res`

