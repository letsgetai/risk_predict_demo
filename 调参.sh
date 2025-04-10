#!/bin/bash

# 定义参数范围
batch_sizes=(32 64 128)  # 批次大小
learning_rates=(0.0001 0.0005 0.001)  # 学习率
num_epochs=(20 30 50)  # 训练轮数
hidden_sizes=(32 64 128)  # 隐藏层大小

# 循环遍历不同参数组合
for batch_size in "${batch_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
        for num_epoch in "${num_epochs[@]}"; do
            for hidden_size in "${hidden_sizes[@]}"; do
                # 生成唯一的结果路径
                result_file="risk_control/results/results_batch_${batch_size}_lr_${learning_rate}_epochs_${num_epoch}_hidden_${hidden_size}.csv"

                # 执行训练脚本
                python risk_control/run.py \
                    --result_path "$result_file" \
                    --batch_size "$batch_size" \
                    --learning_rate "$learning_rate" \
                    --num_epochs "$num_epoch" \
                    --hidden_size "$hidden_size"
                    
                echo "Finished training with batch_size=${batch_size}, learning_rate=${learning_rate}, num_epochs=${num_epoch}, hidden_size=${hidden_size}. Results saved to $result_file."
            done
        done
    done
done
