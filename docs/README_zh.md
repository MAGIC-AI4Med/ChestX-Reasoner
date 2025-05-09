# ChestX-Reasoner: 胸部X光推理基础模型

*ChestX-Reasoner: 通过逐步验证推理提升放射学基础模型*

<p align="center">
    <a href="https://arxiv.org/pdf/2504.20930"><img src="https://img.shields.io/badge/📄-论文-red"></a>
    <a href="tobe completed"><img src="https://img.shields.io/badge/🤗 HuggingFace-数据与模型-green"></a>
</p>



# 安装 
```bash

```
# 数据
```bash

```
# 评估
```bash
cd eval
```
目录说明：
```bash
eval/inference: baseline推理代码
eval/output: baseline推理的预测结果
eval/res: baseline的评估结果
    eval/res/accuracy_res: 准确性评估结果
    eval/res/reasoning_res: 推理评估结果
```

使用手册：
1. 你需要首先根据相关baseline的官方文档配置模型的环境
2. 下载对应的模型权重，将code中的`/path/to/your/model/checkpoint`路径替换为本地的路径
3. 在gpt4o.py中，配置你的`api_key`和`base_url`


### 推理评测：
`dataset.json`中配置了所有评测所需要的json文件路径，可以根据需要进行评测：
```bash
conda activate xxx
python reasoning_eval.py --model xxx --task xxx
```
输出的结果文件会在`eval/output`文件夹下

### 准确性评测：
```bash
python accuracy_eval.py 
```
评测的模型和数据集需要在code中指定