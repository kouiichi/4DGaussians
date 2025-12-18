#!/usr/bin/env python
"""
GMFlow 光流测试脚本
从 toyarm_tiny 数据集读取连续两帧图像，计算光流并可视化

Usage:
    cd /home/ubuntu/zj/4DGaussians
    python scripts/test_gmflow.py --data_path /home/ubuntu/project/data/toyarm_tiny
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmflow.gmflow import build_gmflow
from gmflow.config import get_cfg as get_gmflow_cfg


def load_image(path):
    """加载图像并转换为 tensor"""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    return img


def flow_to_color(flow, max_flow=None):
    """
    将光流转换为颜色可视化
    
    Args:
        flow: [2, H, W] 光流，flow[0] 是 u (水平)，flow[1] 是 v (垂直)
        max_flow: 归一化用的最大流量值
    
    Returns:
        [H, W, 3] RGB 图像
    """
    flow = flow.cpu().numpy() if torch.is_tensor(flow) else flow
    
    u = flow[0]
    v = flow[1]
    
    # 计算光流大小和方向
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    # 归一化
    if max_flow is None:
        max_flow = np.max(magnitude)
    if max_flow > 0:
        magnitude = magnitude / max_flow
    
    # 使用 HSV 颜色空间
    # H: 方向 (0-180)
    # S: 饱和度 (固定为1)
    # V: 亮度 (大小)
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # H
    hsv[..., 1] = 255  # S
    hsv[..., 2] = (np.clip(magnitude, 0, 1) * 255).astype(np.uint8)  # V
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def draw_flow_arrows(img, flow, step=20, scale=1.0):
    """
    在图像上绘制光流箭头
    
    Args:
        img: [H, W, 3] RGB 图像
        flow: [2, H, W] 光流
        step: 箭头采样间隔
        scale: 箭头缩放因子
    """
    flow = flow.cpu().numpy() if torch.is_tensor(flow) else flow
    img = img.copy()
    
    h, w = flow.shape[1], flow.shape[2]
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx = flow[0, y, x] * scale
            dy = flow[1, y, x] * scale
            
            # 只绘制有意义的光流
            if np.sqrt(dx**2 + dy**2) > 0.5:
                cv2.arrowedLine(img, 
                               (x, y), 
                               (int(x + dx), int(y + dy)),
                               (0, 255, 0), 
                               1, 
                               tipLength=0.3)
    
    return img


def create_flow_legend():
    """创建光流颜色图例"""
    size = 100
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # 创建单位圆内的光流
    flow = np.stack([xx, yy], axis=0)
    mask = (xx**2 + yy**2) <= 1
    
    legend = flow_to_color(flow, max_flow=1.0)
    legend[~mask] = 255  # 圆外设为白色
    
    return legend


def main():
    parser = argparse.ArgumentParser(description='Test GMFlow on toyarm dataset')
    parser.add_argument('--data_path', type=str, 
                        default='/home/ubuntu/project/data/toyarm_tiny',
                        help='Path to toyarm dataset')
    parser.add_argument('--frame1', type=int, default=0, 
                        help='First frame index')
    parser.add_argument('--frame2', type=int, default=1, 
                        help='Second frame index')
    parser.add_argument('--camera_idx', type=int, default=0,
                        help='Camera index to use')
    parser.add_argument('--output_dir', type=str, default='./output/flow_debug',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集信息
    transforms_path = os.path.join(args.data_path, 'transforms.json')
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    print(f"Loaded dataset with {len(transforms['frames'])} frames")
    
    # 按时间排序帧
    frames = sorted(transforms['frames'], key=lambda x: x.get('time', 0))
    
    # 按相机分组
    camera_frames = {}
    for frame in frames:
        cam_idx = frame.get('camera_idx', 0)
        if cam_idx not in camera_frames:
            camera_frames[cam_idx] = []
        camera_frames[cam_idx].append(frame)
    
    print(f"Found {len(camera_frames)} cameras")
    for cam_idx, cam_frames in camera_frames.items():
        print(f"  Camera {cam_idx}: {len(cam_frames)} frames")
    
    # 选择相机和帧
    if args.camera_idx not in camera_frames:
        print(f"Camera {args.camera_idx} not found, using camera 0")
        args.camera_idx = list(camera_frames.keys())[0]
    
    cam_frames = camera_frames[args.camera_idx]
    
    if args.frame2 >= len(cam_frames):
        args.frame2 = len(cam_frames) - 1
        args.frame1 = max(0, args.frame2 - 1)
    
    frame1_info = cam_frames[args.frame1]
    frame2_info = cam_frames[args.frame2]
    
    # 加载图像
    img1_path = os.path.join(args.data_path, frame1_info['file_path'])
    img2_path = os.path.join(args.data_path, frame2_info['file_path'])
    
    # 处理路径（可能有 ./ 前缀）
    if img1_path.startswith('./'):
        img1_path = os.path.join(args.data_path, img1_path[2:])
    if img2_path.startswith('./'):
        img2_path = os.path.join(args.data_path, img2_path[2:])
    
    print(f"\nLoading images:")
    print(f"  Frame 1: {img1_path}")
    print(f"  Frame 2: {img2_path}")
    
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    
    print(f"  Image shape: {img1.shape}")
    
    # 打印控制向量信息
    if 'joint_positions' in frame1_info:
        print(f"\nJoint positions:")
        print(f"  Frame 1: {frame1_info['joint_positions']}")
        print(f"  Frame 2: {frame2_info['joint_positions']}")
    
    # 加载 GMFlow
    print("\nLoading GMFlow...")
    cfg = get_gmflow_cfg()
    flownet = torch.nn.DataParallel(build_gmflow(cfg))
    flownet = flownet.module
    
    checkpoint = torch.load(cfg.model, map_location='cpu')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flownet.load_state_dict(weights)
    flownet = flownet.cuda()
    flownet.eval()
    print("GMFlow loaded successfully!")
    
    # 计算光流
    print("\nComputing optical flow...")
    with torch.no_grad():
        # GMFlow 期望输入 [B, C, H, W]，值范围 [0, 255]
        img1_batch = img1.unsqueeze(0).cuda()  # [1, 3, H, W]
        img2_batch = img2.unsqueeze(0).cuda()
        
        print(f"  Input shape: {img1_batch.shape}")
        print(f"  Input range: [{img1_batch.min():.1f}, {img1_batch.max():.1f}]")
        
        flow_preds = flownet(img1_batch, img2_batch)
        flow = flow_preds[0].squeeze(0)  # [2, H, W]
    
    print(f"  Output flow shape: {flow.shape}")
    
    # 光流统计
    flow_np = flow.cpu().numpy()
    u, v = flow_np[0], flow_np[1]
    magnitude = np.sqrt(u**2 + v**2)
    
    print(f"\nFlow statistics:")
    print(f"  U (horizontal): min={u.min():.2f}, max={u.max():.2f}, mean={u.mean():.2f}")
    print(f"  V (vertical):   min={v.min():.2f}, max={v.max():.2f}, mean={v.mean():.2f}")
    print(f"  Magnitude:      min={magnitude.min():.2f}, max={magnitude.max():.2f}, mean={magnitude.mean():.2f}")
    
    # 可视化
    print(f"\nGenerating visualizations...")
    
    # 1. 原始图像
    img1_np = img1.permute(1, 2, 0).numpy().astype(np.uint8)
    img2_np = img2.permute(1, 2, 0).numpy().astype(np.uint8)
    
    # 2. 光流颜色可视化
    flow_color = flow_to_color(flow)
    
    # 3. 带箭头的光流
    flow_arrows = draw_flow_arrows(img1_np, flow, step=30, scale=2.0)
    
    # 4. 光流大小热力图
    magnitude_normalized = (magnitude / (magnitude.max() + 1e-7) * 255).astype(np.uint8)
    magnitude_heatmap = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
    magnitude_heatmap = cv2.cvtColor(magnitude_heatmap, cv2.COLOR_BGR2RGB)
    
    # 5. 颜色图例
    legend = create_flow_legend()
    
    # 创建综合可视化图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title(f'Frame {args.frame1}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title(f'Frame {args.frame2}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(flow_color)
    axes[0, 2].set_title('Optical Flow (Color)')
    axes[0, 2].axis('off')
    
    # 第二行
    axes[1, 0].imshow(flow_arrows)
    axes[1, 0].set_title('Flow Arrows')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(magnitude_heatmap)
    axes[1, 1].set_title(f'Flow Magnitude (max={magnitude.max():.1f}px)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(legend)
    axes[1, 2].set_title('Color Legend\n(center=0, edge=max)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(args.output_dir, f'flow_cam{args.camera_idx}_f{args.frame1}_f{args.frame2}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # 保存单独的光流图
    cv2.imwrite(os.path.join(args.output_dir, 'flow_color.png'), 
                cv2.cvtColor(flow_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.output_dir, 'flow_magnitude.png'), 
                cv2.cvtColor(magnitude_heatmap, cv2.COLOR_RGB2BGR))
    
    # 保存光流数据
    np.save(os.path.join(args.output_dir, 'flow.npy'), flow_np)
    print(f"Saved flow data to: {os.path.join(args.output_dir, 'flow.npy')}")
    
    plt.show()
    print("\nDone!")


if __name__ == '__main__':
    main()
