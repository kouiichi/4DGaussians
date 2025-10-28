#!/usr/bin/env python3
"""
将Neural-Jacobian-Field (NJF) ToyArm数据集格式转换为4DGaussians可用格式

用法:
    python convert_njf_to_4dgs.py --input /path/to/njf/transforms.json --output /path/to/4dgs/data
"""

import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert NJF ToyArm dataset to 4DGaussians format')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to NJF transforms.json file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for 4DGaussians dataset')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Fraction of data to use for training (default: 0.9)')
    parser.add_argument('--camera_angle_x', type=float, default=None,
                        help='Override camera_angle_x if provided (in radians)')
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images to output directory (otherwise use relative paths)')
    parser.add_argument('--use_depth', action='store_true',
                        help='Generate point cloud from depth maps (experimental)')
    parser.add_argument('--coord_transform', choices=['none', 'opencv_to_opengl'], 
                        default='none',
                        help='Coordinate system transformation')
    return parser.parse_args()


def convert_camera_coordinates(transform_matrix: np.ndarray, 
                               transform_type: str = 'none') -> np.ndarray:
    """
    转换相机坐标系
    
    Args:
        transform_matrix: 4x4相机变换矩阵
        transform_type: 转换类型
            - 'none': 不转换
            - 'opencv_to_opengl': OpenCV坐标系转OpenGL/Blender坐标系
    
    Returns:
        转换后的4x4矩阵
    """
    matrix = np.array(transform_matrix, dtype=np.float32)
    
    if transform_type == 'opencv_to_opengl':
        # OpenCV: X右, Y下, Z前
        # OpenGL: X右, Y上, Z后
        # 需要翻转Y和Z轴
        matrix[:3, 1:3] *= -1
    
    return matrix.tolist()


def calculate_camera_angle_x(fl_x: float, width: int) -> float:
    """
    从焦距和图像宽度计算camera_angle_x (FOV)
    
    Args:
        fl_x: X方向焦距(像素)
        width: 图像宽度(像素)
    
    Returns:
        camera_angle_x in radians
    """
    return 2 * np.arctan(width / (2 * fl_x))


def load_njf_transforms(json_path: Path) -> Dict:
    """加载NJF格式的transforms.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'cameras' not in data or 'frames' not in data:
        raise ValueError("Invalid NJF format: missing 'cameras' or 'frames' field")
    
    return data


def merge_camera_and_frame(frame: Dict, cameras: List[Dict], 
                           coord_transform: str = 'none',
                           prev_joint_pos: np.ndarray = None) -> Dict:
    """
    将相机参数合并到帧数据中
    
    Args:
        frame: 帧数据字典
        cameras: 相机参数列表
        coord_transform: 坐标系转换类型
        prev_joint_pos: 上一帧的关节位置（用于计算差值u）
    
    Returns:
        合并后的帧数据（4DGaussians格式）
    """
    camera_idx = frame.get('camera_idx', 0)
    
    if camera_idx >= len(cameras):
        raise ValueError(f"Camera index {camera_idx} out of range (max: {len(cameras)-1})")
    
    camera = cameras[camera_idx]
    
    # 提取并转换transform_matrix
    transform_matrix = convert_camera_coordinates(
        camera['transform_matrix'], 
        coord_transform
    )
    
    # 构建4DGaussians格式的帧数据
    merged_frame = {
        'file_path': frame['file_path'],
        'transform_matrix': transform_matrix,
        'time': frame['time']
    }
    
    # 计算关节位置差值 u = current_joint_pos - prev_joint_pos
    if 'joint_pos' in frame:
        curr_joint_pos = np.array(frame['joint_pos'])
        
        if prev_joint_pos is not None:
            # 计算差值
            u = (curr_joint_pos - prev_joint_pos).tolist()
        else:
            # 第一帧，没有前一帧，u设为0向量
            u = np.zeros_like(curr_joint_pos).tolist()
        
        merged_frame['u'] = u
        merged_frame['joint_pos'] = frame['joint_pos']  # 保留原始关节位置供调试
    
    # 可选：添加额外信息（4DGaussians可能不使用，但保留以便调试）
    if 'depth_file_path' in frame:
        merged_frame['depth_file_path'] = frame['depth_file_path']
    if 'sample_idx' in frame:
        merged_frame['sample_idx'] = frame['sample_idx']
    if 'camera_idx' in frame:
        merged_frame['camera_idx'] = frame['camera_idx']
    
    return merged_frame


def split_train_test(frames: List[Dict], train_split: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
    """
    将帧数据划分为训练集和测试集
    
    策略：按时间戳分组，然后按sample_idx划分
    这样可以确保测试集包含完整的时间序列
    """
    # 按sample_idx分组
    samples = {}
    for frame in frames:
        sample_idx = frame.get('sample_idx', 0)
        if sample_idx not in samples:
            samples[sample_idx] = []
        samples[sample_idx].append(frame)
    
    # 计算训练集样本数
    num_samples = len(samples)
    num_train = int(num_samples * train_split)
    
    # 划分
    train_frames = []
    test_frames = []
    
    for idx, (sample_idx, sample_frames) in enumerate(sorted(samples.items())):
        if idx < num_train:
            train_frames.extend(sample_frames)
        else:
            test_frames.extend(sample_frames)
    
    return train_frames, test_frames


def generate_4dgs_transforms(njf_data: Dict, 
                             train_frames: List[Dict],
                             test_frames: List[Dict],
                             coord_transform: str = 'none',
                             camera_angle_x_override: float = None) -> Tuple[Dict, Dict]:
    """
    生成4DGaussians格式的transforms_train.json和transforms_test.json
    
    注意：会按时间顺序处理帧，以正确计算关节位置差值u
    """
    cameras = njf_data['cameras']
    
    # 使用第一个相机的参数计算camera_angle_x
    camera0 = cameras[0]
    
    if camera_angle_x_override is not None:
        camera_angle_x = camera_angle_x_override
    elif 'camera_angle_x' in camera0:
        camera_angle_x = camera0['camera_angle_x']
    else:
        camera_angle_x = calculate_camera_angle_x(
            camera0['fl_x'], 
            camera0['w']
        )
    
    # 处理训练集：按时间和camera_idx排序，以正确计算u
    train_frames_sorted = sorted(train_frames, key=lambda x: (x.get('time', 0), x.get('camera_idx', 0)))
    train_frames_with_u = []
    prev_joint_pos_per_camera = {}  # 为每个相机维护独立的前一帧关节位置
    
    for frame in train_frames_sorted:
        camera_idx = frame.get('camera_idx', 0)
        prev_joint_pos = prev_joint_pos_per_camera.get(camera_idx, None)
        
        merged_frame = merge_camera_and_frame(frame, cameras, coord_transform, prev_joint_pos)
        train_frames_with_u.append(merged_frame)
        
        # 更新该相机的前一帧关节位置
        if 'joint_pos' in frame:
            prev_joint_pos_per_camera[camera_idx] = np.array(frame['joint_pos'])
    
    # 处理测试集：同样的逻辑
    test_frames_sorted = sorted(test_frames, key=lambda x: (x.get('time', 0), x.get('camera_idx', 0)))
    test_frames_with_u = []
    prev_joint_pos_per_camera = {}  # 重置，测试集独立计算
    
    for frame in test_frames_sorted:
        camera_idx = frame.get('camera_idx', 0)
        prev_joint_pos = prev_joint_pos_per_camera.get(camera_idx, None)
        
        merged_frame = merge_camera_and_frame(frame, cameras, coord_transform, prev_joint_pos)
        test_frames_with_u.append(merged_frame)
        
        if 'joint_pos' in frame:
            prev_joint_pos_per_camera[camera_idx] = np.array(frame['joint_pos'])
    
    # 构建训练集transforms
    transforms_train = {
        'camera_angle_x': camera_angle_x,
        'fl_x': camera0.get('fl_x'),
        'fl_y': camera0.get('fl_y'),
        'cx': camera0.get('cx'),
        'cy': camera0.get('cy'),
        'w': camera0.get('w'),
        'h': camera0.get('h'),
        'frames': train_frames_with_u
    }
    
    # 构建测试集transforms
    transforms_test = {
        'camera_angle_x': camera_angle_x,
        'fl_x': camera0.get('fl_x'),
        'fl_y': camera0.get('fl_y'),
        'cx': camera0.get('cx'),
        'cy': camera0.get('cy'),
        'w': camera0.get('w'),
        'h': camera0.get('h'),
        'frames': test_frames_with_u
    }
    
    return transforms_train, transforms_test


def copy_images_if_needed(frames: List[Dict], input_root: Path, output_root: Path):
    """
    如果需要，复制图像文件到输出目录
    """
    print("Copying images...")
    for frame in tqdm(frames):
        src_path = input_root / frame['file_path']
        dst_path = output_root / frame['file_path']
        
        if not src_path.exists():
            print(f"Warning: Source image not found: {src_path}")
            continue
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def generate_initial_point_cloud(output_dir: Path, num_points: int = 2000):
    """
    生成初始点云（简单的随机点云）
    如果需要使用depth数据生成点云，这里可以扩展
    """
    print(f"Generating initial point cloud with {num_points} points...")
    
    # 生成随机点云
    xyz = np.random.random((num_points, 3)) * 2.6 - 1.3
    rgb = np.random.random((num_points, 3))
    
    # 保存为PLY格式
    ply_path = output_dir / "fused.ply"
    
    # PLY header
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(ply_path, 'w') as f:
        f.write(header)
        for i in range(num_points):
            r, g, b = (rgb[i] * 255).astype(np.uint8)
            f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} 0 0 0 {r} {g} {b}\n")
    
    print(f"Point cloud saved to: {ply_path}")


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # 验证输入
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading NJF transforms from: {input_path}")
    njf_data = load_njf_transforms(input_path)
    
    print(f"Found {len(njf_data['cameras'])} cameras and {len(njf_data['frames'])} frames")
    
    # 划分训练集和测试集
    print(f"Splitting data with train_split={args.train_split}")
    train_frames, test_frames = split_train_test(njf_data['frames'], args.train_split)
    print(f"Train frames: {len(train_frames)}, Test frames: {len(test_frames)}")
    
    # 生成4DGaussians格式的transforms
    print("Generating 4DGaussians transforms...")
    transforms_train, transforms_test = generate_4dgs_transforms(
        njf_data,
        train_frames,
        test_frames,
        coord_transform=args.coord_transform,
        camera_angle_x_override=args.camera_angle_x
    )
    
    # 保存transforms文件
    train_json_path = output_dir / "transforms_train.json"
    test_json_path = output_dir / "transforms_test.json"
    
    print(f"Saving transforms_train.json to: {train_json_path}")
    with open(train_json_path, 'w') as f:
        json.dump(transforms_train, f, indent=4)
    
    print(f"Saving transforms_test.json to: {test_json_path}")
    with open(test_json_path, 'w') as f:
        json.dump(transforms_test, f, indent=4)
    
    # 复制图像（如果需要）
    if args.copy_images:
        input_root = input_path.parent
        all_frames = train_frames + test_frames
        copy_images_if_needed(all_frames, input_root, output_dir)
    else:
        print("Skipping image copy (use --copy_images to enable)")
    
    # 生成初始点云
    generate_initial_point_cloud(output_dir)
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - transforms_train.json: {len(train_frames)} frames")
    print(f"  - transforms_test.json: {len(test_frames)} frames")
    print(f"  - fused.ply: Initial point cloud")
    
    # 检查并显示u的统计信息
    if transforms_train['frames'] and 'u' in transforms_train['frames'][0]:
        print(f"\n✅ Joint position differences (u) computed successfully!")
        print(f"  - u dimension: {len(transforms_train['frames'][0]['u'])}")
        
        # 显示一个示例
        example_frame = transforms_train['frames'][1] if len(transforms_train['frames']) > 1 else transforms_train['frames'][0]
        print(f"  - Example u: {[f'{v:.3f}' for v in example_frame['u']]}")
        
        # 计算u的统计
        all_u = np.array([f['u'] for f in transforms_train['frames'] if 'u' in f])
        print(f"  - u statistics:")
        print(f"    Mean: {np.mean(all_u, axis=0)}")
        print(f"    Std:  {np.std(all_u, axis=0)}")
        print(f"    Max:  {np.max(np.abs(all_u), axis=0)}")
    
    print("\nNext steps:")
    print(f"  1. Review the generated files in {output_dir}")
    if not args.copy_images:
        print(f"  2. Make sure images are accessible from {output_dir}")
        print(f"     (or rerun with --copy_images to copy them)")
    print(f"  3. Run 4DGaussians training:")
    print(f"     python train.py -s {output_dir} -m output/toyarm --eval")


if __name__ == '__main__':
    main()
