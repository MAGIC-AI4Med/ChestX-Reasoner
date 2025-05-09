#!/bin/bash

# 获取当前时间，格式为 YYYYMMDD_HHMMSS
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# 定义模型列表
MODELS=(
    # "llavanext" 
    # "deepseek"
    # "gpt4o"
    # "ours"
    # "qwenvl7b" 
    # "radfm"
    # "meddr"
    # "qwenvl72b"
    # "ours_rl50"
    # "ours_process"
    # "chexagent8b"
    # "chexagent3b"
    # "ours_process_final_150"
    # "ours_process_final_240"
    # "ours_process_final_160"
    # "ours_reason_final_150"
    # "ours_cold_final_300"

    #ablation
    # "ours_rl0"
    # "ours_rl1"
    # "ours_sft0"
    # "ours_sft1"
    "ours_process_300"
    # "ours_process_wo"
)

# 定义任务列表
TASKS=(
    "chexpert_multi_no_option"
    "chexpert_binary"
    "chexpert_multi"
    "chexpert_single"
    
    "mimic_multi_no_option"
    "mimic_single"
    "mimic_binary"
    "mimic_multi"
    
    "temporal"
    
    
    #舍弃
    # "openi_single"
    # "openi_multi"
)
# 定义起始版本和结束版本
START_VERSION=19
END_VERSION=19


# 遍历所有模型和任务
for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for VERSION in $(seq $START_VERSION $END_VERSION); do
            echo "Evaluating model: $MODEL on task: $TASK"

            # 创建日志目录
            LOG_DIR="/mnt/petrelfs/radshare/reasonbench/VRBench/log/${CURRENT_TIME}/${MODEL}"
            mkdir -p "$LOG_DIR"

            # 提交评估任务
            srun -p medai --time=10-00:00:00 --quotatype=auto \
                --job-name eval_${MODEL}_${TASK} --cpus-per-task=16 \
                /mnt/petrelfs/fanziqing/.conda/envs/radfm/bin/python \
                /mnt/petrelfs/radshare/reasonbench/VRBench/asseess.py \
                --model $MODEL \
                --task $TASK > "${LOG_DIR}/${TASK}_output.log" \
                --id $VERSION 2>&1 &
            
            # 每个任务间隔1秒
            sleep 1
        done
    done
done

# 等待所有任务完成
wait
echo "All evaluation jobs completed."