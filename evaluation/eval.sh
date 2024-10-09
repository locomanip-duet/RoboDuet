#!/bin/bash

# 基础配置
BASE_CMD="python test_tracking_sample_error.py --sample_dir samples --headless --stand_still --num_envs 5000 --sim_device cuda:0"

# 定义检查点数组
CHECKPOINTS=(49200 49600 50000)

# Baseline 实验
for seed in 5078 8765; do
    for ckpt in "${CHECKPOINTS[@]}"; do
        $BASE_CMD --ckpt_folder /home/a4090/3party/ablation/baseline/$seed --net_type U -ckptn $ckpt --output_file "baseline_seed${seed}_ckpt${ckpt}.log"
    done
done

# RoboDuet_926 实验
for seed in 4295 5078 2423; do
    folder=$(ls -d /home/a4090/3party/ablation/RoboDuet_926/*seed$seed)
    for ckpt in "${CHECKPOINTS[@]}"; do
        $BASE_CMD --ckpt_folder $folder --net_type C -ckptn $ckpt --output_file "roboduet_seed${seed}_ckpt${ckpt}.log"
    done
done

# Cooperate 实验
for seed in 2423 8765 5078; do
    folder=$(ls -d /home/a4090/3party/ablation/Cooperate/*seed$seed)
    for ckpt in "${CHECKPOINTS[@]}"; do
        $BASE_CMD --ckpt_folder $folder --net_type C -ckptn $ckpt --output_file "cooperate_seed${seed}_ckpt${ckpt}.log"
    done
done

# Twostage 实验
for seed in 2423 5078 8765; do
    for ckpt in "${CHECKPOINTS[@]}"; do
        $BASE_CMD --ckpt_folder /home/a4090/3party/ablation/twostage/$seed --net_type U -ckptn $ckpt --output_file "twostage_seed${seed}_ckpt${ckpt}.log"
    done
done