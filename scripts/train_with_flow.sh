#!/bin/bash

# Example training script with optical flow supervision and improved visualization
# 使用光流监督并增强可视化功能的训练示例脚本

# GPU设置 - 使用第三个GPU（GPU 2）
export CUDA_VISIBLE_DEVICES=2

SOURCE_PATH="/home/ubuntu/project/data/toyarm_tiny"
BASE_OUTPUT_DIR="/home/ubuntu/project/outputs"

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 使用时间戳的模型路径
MODEL_PATH="${BASE_OUTPUT_DIR}/flow_${TIMESTAMP}"

# 光流可视化配置
FLOW_VIS_DIR="${MODEL_PATH}/flow_vis"  # 光流可视化目录
FLOW_VIS_INTERVAL=1000  # 光流生成间隔（每50次迭代生成一次，之前是100次）

# 训练参数
ITERATIONS=15000
COARSE_ITERATIONS=3000

# 光流监督参数
USE_FLOW_LOSS="--use_flow_loss"  # 启用光流监督
LAMBDA_FLOW=0.005                  # 光流loss权重
FLOW_START_ITER=3000             # 从3000次迭代开始使用光流

# 其他参数
BATCH_SIZE=4

# 创建输出目录
mkdir -p ${MODEL_PATH}
mkdir -p ${FLOW_VIS_DIR}

echo "Training will save models to: ${MODEL_PATH}"
echo "Optical flow visualizations will be saved to: ${FLOW_VIS_DIR}"

# 运行训练
python train.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --iterations ${ITERATIONS} \
    --coarse_iterations ${COARSE_ITERATIONS} \
    ${USE_FLOW_LOSS} \
    --lambda_flow ${LAMBDA_FLOW} \
    --flow_loss_start_iter ${FLOW_START_ITER} \
    --batch_size ${BATCH_SIZE} \
    --eval \
    --flow_vis_dir ${FLOW_VIS_DIR} \
    --flow_vis_interval ${FLOW_VIS_INTERVAL}

echo "Training with flow supervision completed!"
