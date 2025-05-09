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
_**Ziqing Fan<sup>1,2 </sup>, Cheng Liang<sup>1,2 </sup>, Chaoyi Wu<sup>1</sup>, Ya Zhang<sup>1,2</sup>, Yanfeng Wang<sup>1,2</sup>, Weidi Xie<sup>1,2</sup>**_


<!-- **Affiliations:** -->

_<sup>1</sup> Shanghai Jiao Tong University,
<sup>2</sup> Shanghai AI Laboratory._

</div>

The official codes for "ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification".  

## Usage
### Environment  


### Supervised Fine-Tuning  
We provide all the code used for further training on MMedC. The codes are in the `pretrain` folder. You can check the [documentation](./pretrain/README.md) in the folder for how to use the codes.

* Note that this step requires at least 8 A100 80GB GPUs and training for over a month.

### Reinforcement Learning  
We provide all the code used for fine-tuning. We support 2 fine-tuning methods: Full-Model Fine-tuning and PEFT Fine-Tuning.  Both codes are in the `finetune` folder. You can check the [documentation](./finetune/README.md) in the folder for how to use the codes.

### Reinforcement Learning with Process Reward  

### Benchmark Data  
In `eval/data`, we present our benchmark construction code and our data.
### Evaluation  
We provide:
1. The evaluation code on both reasoning and accuracy in `eval/`
2. The baseline inference code in `eval/inference`
3. The evaluation results on both reasoning and accuracy of all baselines in `eval/res`