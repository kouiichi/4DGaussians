"""
NJF (Neural-Jacobian-Field) ToyArm数据集加载器
为4DGaussians提供直接加载NJF格式数据的能力

使用方法：
1. 将此文件放入 scene/ 目录
2. 在 scene/dataset_readers.py 中添加相应的导入和注册
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import NamedTuple, List, Dict, Optional
from scene.dataset_readers import (
    CameraInfo, 
    SceneInfo, 
    BasicPointCloud,
    getNerfppNorm,
    storePly,
    fetchPly
)
from utils.graphics_utils import focal2fov
from utils.general_utils import PILtoTorch
from utils.sh_utils import SH2RGB


class CameraInfoWithU(NamedTuple):
    """扩展的CameraInfo，包含关节位置差值u"""
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    mask: np.array
    u: Optional[np.array]  # 关节位置差值
    joint_pos: Optional[List[float]]  # 原始关节位置（用于调试）


def readNJFTransforms(path: str, transformsfile: str = "transforms.json", 
                      white_background: bool = False) -> tuple:
    """
    读取NJF格式的transforms文件
    
    Args:
        path: 数据集根目录
        transformsfile: transforms文件名（默认"transforms.json"）
        white_background: 是否使用白色背景
    
    Returns:
        (cam_infos, cameras)
        cam_infos中每个CameraInfo会包含计算好的u（关节位置差值）
    """
    json_path = os.path.join(path, transformsfile)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'cameras' not in data or 'frames' not in data:
        raise ValueError(f"Invalid NJF format in {json_path}: missing 'cameras' or 'frames'")
    
    cameras = data['cameras']
    frames = data['frames']
    
    print(f"Loading NJF dataset: {len(cameras)} cameras, {len(frames)} frames")
    
    # 按时间和camera_idx排序，以正确计算关节位置差值
    frames_sorted = sorted(frames, key=lambda x: (x.get('time', 0), x.get('camera_idx', 0)))
    
    # 为每个相机维护前一帧的关节位置
    prev_joint_pos_per_camera = {}
    
    # 构建相机信息列表
    cam_infos = []
    
    for idx, frame in enumerate(frames_sorted):
        camera_idx = frame.get('camera_idx', 0)
        
        if camera_idx >= len(cameras):
            print(f"Warning: camera_idx {camera_idx} out of range, skipping frame {idx}")
            continue
        
        camera = cameras[camera_idx]
        
        # 读取图像
        image_path = os.path.join(path, frame['file_path'])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}, skipping")
            continue
        
        image_name = Path(frame['file_path']).stem
        image = Image.open(image_path)
        
        # 处理背景
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        
        if im_data.shape[2] == 4:  # 有alpha通道
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        else:
            image = Image.fromarray(im_data[:, :, :3], "RGB")
        
        # 转换为tensor（4DGaussians使用固定尺寸800x800）
        image = PILtoTorch(image, (800, 800))
        
        # 提取相机参数
        fl_x = camera.get('fl_x', camera.get('focal_x', 606.0))
        fl_y = camera.get('fl_y', camera.get('focal_y', 606.0))
        width = camera.get('w', 640)
        height = camera.get('h', 480)
        
        # 计算FOV
        FovX = focal2fov(fl_x, width)
        FovY = focal2fov(fl_y, height)
        
        # 提取并转换相机外参
        c2w = np.array(camera['transform_matrix'], dtype=np.float32)
        
        # 转换为world-to-camera (w2c)
        w2c = np.linalg.inv(c2w)
        
        # 提取R和T（4DGaussians的约定）
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        # 提取时间戳（已归一化到0-1）
        time = frame.get('time', 0.0)
        
        # 计算关节位置差值 u
        joint_pos_diff = None
        if 'joint_pos' in frame:
            curr_joint_pos = np.array(frame['joint_pos'])
            prev_joint_pos = prev_joint_pos_per_camera.get(camera_idx, None)
            
            if prev_joint_pos is not None:
                joint_pos_diff = curr_joint_pos - prev_joint_pos
            else:
                # 第一帧，设为零向量
                joint_pos_diff = np.zeros_like(curr_joint_pos)
            
            # 更新该相机的前一帧关节位置
            prev_joint_pos_per_camera[camera_idx] = curr_joint_pos
        
        # 创建CameraInfo（扩展版，包含u）
        cam_info = CameraInfoWithU(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=image.shape[1],
            height=image.shape[2],
            time=time,
            mask=None,
            u=joint_pos_diff,  # 添加关节位置差值
            joint_pos=frame.get('joint_pos', None)  # 保留原始关节位置供调试
        )
        
        cam_infos.append(cam_info)
    
    return cam_infos, cameras


def readNJFSceneInfo(path: str, white_background: bool = False, eval: bool = True, 
                     train_test_split: float = 0.9) -> SceneInfo:
    """
    读取NJF格式的场景信息
    
    Args:
        path: 数据集根目录
        white_background: 是否使用白色背景
        eval: 是否划分训练/测试集
        train_test_split: 训练集比例（默认0.9）
    
    Returns:
        SceneInfo对象
    """
    print(f"Reading NJF dataset from: {path}")
    
    # 读取相机信息
    cam_infos, cameras = readNJFTransforms(path, "transforms.json", white_background)
    
    # 按sample_idx分组（NJF的多时间步多视角结构）
    samples = {}
    for cam_info in cam_infos:
        # 从image_name中提取sample_idx（假设格式为：XXXXX_XXXXX）
        # 或者从原始transforms中提取
        # 这里简化处理：按时间分组
        time_key = int(cam_info.time * 100)  # 将时间离散化
        if time_key not in samples:
            samples[time_key] = []
        samples[time_key].append(cam_info)
    
    # 划分训练集和测试集
    if eval:
        num_samples = len(samples)
        num_train = int(num_samples * train_test_split)
        
        train_cam_infos = []
        test_cam_infos = []
        
        for idx, (time_key, sample_cams) in enumerate(sorted(samples.items())):
            if idx < num_train:
                train_cam_infos.extend(sample_cams)
            else:
                test_cam_infos.extend(sample_cams)
        
        print(f"Train cameras: {len(train_cam_infos)}")
        print(f"Test cameras: {len(test_cam_infos)}")
        
        # 显示u的统计信息
        if train_cam_infos and hasattr(train_cam_infos[0], 'u') and train_cam_infos[0].u is not None:
            all_u = [cam.u for cam in train_cam_infos if cam.u is not None]
            if all_u:
                all_u = np.array(all_u)
                print(f"\n✅ Joint position differences (u) computed:")
                print(f"  - u dimension: {all_u.shape[1]}")
                print(f"  - Mean: {np.mean(all_u, axis=0)}")
                print(f"  - Std:  {np.std(all_u, axis=0)}")
                print(f"  - Max abs: {np.max(np.abs(all_u), axis=0)}")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        print(f"Total cameras: {len(train_cam_infos)}")
    
    # 计算场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # 处理点云初始化
    ply_path = os.path.join(path, "fused.ply")
    
    if not os.path.exists(ply_path):
        print(f"Point cloud not found, generating random initialization...")
        
        # 生成随机点云
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts} points)...")
        
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, 
            colors=SH2RGB(shs), 
            normals=np.zeros((num_pts, 3))
        )
        
        # 保存点云以便下次使用
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print(f"Loading existing point cloud from: {ply_path}")
        pcd = fetchPly(ply_path)
    
    # 计算最大时间（用于视频生成）
    max_time = max([cam.time for cam in train_cam_infos])
    
    # 构建SceneInfo
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        video_cameras=train_cam_infos,  # 使用训练集相机生成视频
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=max_time
    )
    
    return scene_info


def readNJFSceneInfoWithDepth(path: str, white_background: bool = False, 
                               eval: bool = True) -> SceneInfo:
    """
    读取NJF格式的场景信息，并使用depth数据初始化点云
    
    这是一个高级功能，可以利用NJF提供的深度图生成更好的初始点云
    
    TODO: 实现深度图点云生成
    """
    # 先读取基本场景信息
    scene_info = readNJFSceneInfo(path, white_background, eval)
    
    # TODO: 利用depth_file_path生成更好的初始点云
    print("Warning: Depth-based initialization not yet implemented")
    print("Using random point cloud instead")
    
    return scene_info


# 辅助函数：从depth图生成点云
def depth_to_pointcloud(depth_image: np.ndarray, intrinsics: Dict, 
                       extrinsics: np.ndarray, subsample: int = 10) -> np.ndarray:
    """
    从深度图生成点云
    
    Args:
        depth_image: 深度图 (H, W)
        intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}
        extrinsics: 相机外参 (4, 4) c2w矩阵
        subsample: 下采样率
    
    Returns:
        points: (N, 3) 世界坐标系中的3D点
    """
    h, w = depth_image.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u[::subsample, ::subsample].flatten()
    v = v[::subsample, ::subsample].flatten()
    depth = depth_image[::subsample, ::subsample].flatten()
    
    # 过滤无效深度
    valid = (depth > 0) & (depth < 10.0)
    u, v, depth = u[valid], v[valid], depth[valid]
    
    # 反投影到相机坐标系
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # 转换到世界坐标系
    points_cam = np.stack([x, y, z, np.ones_like(x)], axis=1)
    points_world = (extrinsics @ points_cam.T).T[:, :3]
    
    return points_world
