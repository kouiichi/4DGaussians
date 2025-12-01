"""
Camera Sampler Module for Flow Computation
用于光流计算的相机采样模块

功能：
1. 按相机ID分组相机列表
2. 采样同一相机的连续时间帧对
3. 确保光流计算使用同一视角的时间连续帧
"""

import random
from collections import defaultdict


def group_cameras_by_view(camera_list):
    """
    将相机列表按照视角(colmap_id或uid)分组，并在每个组内按时间排序
    
    Args:
        camera_list: 相机对象列表，每个相机应有以下属性之一：
                    - colmap_id: COLMAP相机ID（优先使用）
                    - uid: 唯一标识符
                    - image_name: 图像文件名（可从中提取相机ID）
                    - time: 时间戳（用于排序）
    
    Returns:
        dict: {camera_id: [sorted cameras by time]}
              相机ID到相机列表的映射，每个列表按时间排序
    
    Example:
        >>> cameras = [cam1, cam2, cam3, ...]
        >>> groups = group_cameras_by_view(cameras)
        >>> # groups = {0: [cam_t0, cam_t1, ...], 1: [cam_t0, cam_t1, ...], ...}
    """
    camera_groups = defaultdict(list)
    
    for cam in camera_list:
        # 尝试使用camera_idx作为相机标识（最可靠）
        if hasattr(cam, 'camera_idx'):
            cam_id = cam.camera_idx
        # 如果没有camera_idx，尝试使用uid
        elif hasattr(cam, 'uid'):
            cam_id = cam.uid
        # 如果都没有，尝试从image_name中提取相机ID
        elif hasattr(cam, 'image_name'):
            cam_id = extract_camera_id_from_name(cam.image_name)
        else:
            # 默认相机ID（所有相机归为一组）
            cam_id = 0
        
        camera_groups[cam_id].append(cam)
    
    # 在每个组内按时间排序
    for cam_id in camera_groups:
        camera_groups[cam_id].sort(key=lambda x: getattr(x, 'time', 0))
    
    return camera_groups


def extract_camera_id_from_name(image_name):
    """
    从图像文件名或路径中提取相机ID
    
    支持的命名格式：
    - view_0/rgb/00001.png -> camera_id=0
    - view_11/rgb/00099.png -> camera_id=11
    - cam01_frame00.png -> camera_id=1
    - camera_03_t00.jpg -> camera_id=3
    
    Args:
        image_name: 图像文件名或路径
    
    Returns:
        int: 提取的相机ID，如果无法提取则返回0
    """
    import re
    
    # 优先匹配 "view_X/" 格式 (用于当前数据集)
    match = re.search(r'view[_-]?(\d+)', image_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 尝试匹配 "cam" 或 "camera" 后跟数字
    match = re.search(r'cam(?:era)?[_-]?(\d+)', image_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 无法提取，返回默认值
    return 0


def sample_consecutive_frames_same_camera(camera_groups, num_pairs=1, time_tolerance=1.5):
    """
    从相机组中采样连续的帧对，确保来自同一相机且时间相近
    
    Args:
        camera_groups: 按相机ID分组的相机字典 (由group_cameras_by_view返回)
        num_pairs: 需要采样的帧对数量
        time_tolerance: 时间容差，相邻帧时间差应小于此值（默认1.5）
                       用于过滤时间跳跃的帧对
    
    Returns:
        list: [(cam_t, cam_t1), ...] 帧对列表
              每个元素是(时刻t的相机, 时刻t+1的相机)元组
    
    Example:
        >>> camera_groups = group_cameras_by_view(cameras)
        >>> frame_pairs = sample_consecutive_frames_same_camera(camera_groups, num_pairs=2)
        >>> for cam_t, cam_t1 in frame_pairs:
        >>>     # cam_t和cam_t1来自同一相机，时间连续
        >>>     flow = compute_flow(cam_t, cam_t1)
    """
    # 找出所有相机中有时间连续帧对的位置
    consecutive_pairs_pool = []
    
    for cam_id, cam_list in camera_groups.items():
        if len(cam_list) < 2:
            continue
        
        # 找出该相机内所有时间连续的帧对
        for i in range(len(cam_list) - 1):
            cam_t = cam_list[i]
            cam_t1 = cam_list[i + 1]
            
            # 检查时间是否连续（在容差范围内）
            if hasattr(cam_t, 'time') and hasattr(cam_t1, 'time'):
                time_diff = abs(cam_t1.time - cam_t.time)
                if time_diff <= time_tolerance:
                    consecutive_pairs_pool.append((cam_t, cam_t1))
            else:
                # 如果没有time属性，假设排序后的相邻帧就是连续的
                consecutive_pairs_pool.append((cam_t, cam_t1))
    
    if not consecutive_pairs_pool:
        return []
    
    # 从池中随机采样
    num_available = len(consecutive_pairs_pool)
    num_to_sample = min(num_pairs, num_available)
    
    frame_pairs = random.sample(consecutive_pairs_pool, num_to_sample)
    
    return frame_pairs


def get_camera_statistics(camera_groups):
    """
    获取相机分组的统计信息
    
    Args:
        camera_groups: 按相机ID分组的相机字典
    
    Returns:
        dict: 统计信息字典，包含：
            - num_cameras: 相机数量
            - total_frames: 总帧数
            - frames_per_camera: 每个相机的帧数
            - consecutive_pairs_available: 可用的连续帧对数量
    """
    stats = {
        'num_cameras': len(camera_groups),
        'total_frames': sum(len(cams) for cams in camera_groups.values()),
        'frames_per_camera': {},
        'consecutive_pairs_available': 0
    }
    
    for cam_id, cam_list in camera_groups.items():
        stats['frames_per_camera'][cam_id] = len(cam_list)
        if len(cam_list) >= 2:
            stats['consecutive_pairs_available'] += len(cam_list) - 1
    
    return stats


def validate_frame_pair(cam_t, cam_t1, strict=True):
    """
    验证帧对是否有效（来自同一相机且时间连续）
    
    Args:
        cam_t: 时刻t的相机
        cam_t1: 时刻t+1的相机
        strict: 是否严格检查时间连续性
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # 检查相机ID是否相同 (优先使用camera_idx)
    if hasattr(cam_t, 'camera_idx') and hasattr(cam_t1, 'camera_idx'):
        if cam_t.camera_idx != cam_t1.camera_idx:
            return False, f"不同相机ID: {cam_t.camera_idx} vs {cam_t1.camera_idx}"
    elif hasattr(cam_t, 'colmap_id') and hasattr(cam_t1, 'colmap_id'):
        if cam_t.colmap_id != cam_t1.colmap_id:
            return False, f"不同colmap_id: {cam_t.colmap_id} vs {cam_t1.colmap_id}"
    
    # 检查时间连续性
    if strict and hasattr(cam_t, 'time') and hasattr(cam_t1, 'time'):
        time_diff = abs(cam_t1.time - cam_t.time)
        if time_diff > 1.5:
            return False, f"时间不连续: {cam_t.time} -> {cam_t1.time} (差值={time_diff})"
        if cam_t1.time <= cam_t.time:
            return False, f"时间倒序: {cam_t.time} -> {cam_t1.time}"
    
    return True, "OK"


def create_camera_sampler(camera_list, time_tolerance=1.5):
    """
    创建相机采样器的便捷函数
    
    Args:
        camera_list: 相机列表
        time_tolerance: 时间容差
    
    Returns:
        tuple: (camera_groups, sampler_function)
    """
    camera_groups = group_cameras_by_view(camera_list)
    
    def sampler(num_pairs=1):
        return sample_consecutive_frames_same_camera(
            camera_groups, 
            num_pairs=num_pairs, 
            time_tolerance=time_tolerance
        )
    
    return camera_groups, sampler
