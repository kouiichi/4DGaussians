"""
混合采样策略 - 用于4DGaussians训练
策略A: 部分样本用于光流计算（连续帧对），部分样本随机采样（保持多样性）
"""
import random
from collections import defaultdict


def sample_mixed_batch(viewpoint_stack, batch_size, flow_pair_ratio=0.5):
    """
    混合采样策略：部分样本采样连续帧对（用于光流），部分样本随机采样
    
    Args:
        viewpoint_stack: 可用的相机列表
        batch_size: batch大小
        flow_pair_ratio: 用于光流的样本比例（默认0.5，即一半用于光流）
    
    Returns:
        list: 采样的相机列表
        
    示例:
        batch_size=4, flow_pair_ratio=0.5
        返回: [cam7_t50, cam7_t51, cam3_t25, cam9_t88]
              前2个是连续帧对，后2个是随机样本
    """
    viewpoint_cams = []
    
    # 计算用于光流的样本数（必须是偶数）
    num_flow_samples = int(batch_size * flow_pair_ratio)
    if num_flow_samples % 2 == 1:
        num_flow_samples -= 1  # 确保是偶数
    
    num_random_samples = batch_size - num_flow_samples
    
    # 1. 采样连续帧对（用于光流）
    if num_flow_samples > 0:
        # 按相机ID分组
        camera_groups = defaultdict(list)
        for cam in viewpoint_stack:
            cam_id = cam.camera_idx if hasattr(cam, 'camera_idx') else cam.colmap_id
            camera_groups[cam_id].append(cam)
        
        # 对每组按时间排序
        for cam_id in camera_groups:
            camera_groups[cam_id].sort(key=lambda x: getattr(x, 'time', 0))
        
        # 找到所有可用的连续帧对
        consecutive_pairs_pool = []
        for cam_id, cam_list in camera_groups.items():
            if len(cam_list) >= 2:
                for i in range(len(cam_list) - 1):
                    cam_t = cam_list[i]
                    cam_t1 = cam_list[i + 1]
                    # 检查时间连续性
                    if hasattr(cam_t, 'time') and hasattr(cam_t1, 'time'):
                        time_diff = abs(cam_t1.time - cam_t.time)
                        if time_diff <= 0.02:  # 时间容差
                            consecutive_pairs_pool.append((cam_t, cam_t1))
        
        # 从池中随机采样连续帧对
        num_pairs_needed = num_flow_samples // 2
        if len(consecutive_pairs_pool) >= num_pairs_needed:
            selected_pairs = random.sample(consecutive_pairs_pool, num_pairs_needed)
            for cam_t, cam_t1 in selected_pairs:
                viewpoint_cams.extend([cam_t, cam_t1])
        else:
            # 如果可用帧对不足，尽可能多地采样
            for cam_t, cam_t1 in consecutive_pairs_pool:
                viewpoint_cams.extend([cam_t, cam_t1])
            # 剩余的用随机采样补充
            num_random_samples += num_flow_samples - len(viewpoint_cams)
    
    # 2. 随机采样剩余样本（保持多样性）
    available_cams = [cam for cam in viewpoint_stack if cam not in viewpoint_cams]
    if len(available_cams) >= num_random_samples:
        random_cams = random.sample(available_cams, num_random_samples)
        viewpoint_cams.extend(random_cams)
    else:
        # 如果不够，就用所有可用的
        viewpoint_cams.extend(available_cams)
    
    return viewpoint_cams


def sample_mixed_batch_from_pool(viewpoint_stack, temp_list, batch_size, flow_pair_ratio=0.5):
    """
    从可弹出的池中进行混合采样（兼容原有的pop机制）
    
    Args:
        viewpoint_stack: 当前可用的相机池（会被修改）
        temp_list: 完整的相机列表（用于补充）
        batch_size: batch大小
        flow_pair_ratio: 用于光流的样本比例
    
    Returns:
        list: 采样的相机列表
    """
    viewpoint_cams = []
    
    # 如果池中样本不足，先补充
    if len(viewpoint_stack) < batch_size:
        viewpoint_stack.extend(temp_list.copy())
    
    # 计算用于光流的样本数（必须是偶数）
    num_flow_samples = int(batch_size * flow_pair_ratio)
    if num_flow_samples % 2 == 1:
        num_flow_samples -= 1
    
    num_random_samples = batch_size - num_flow_samples
    
    # 1. 尝试采样连续帧对
    if num_flow_samples > 0:
        # 按相机ID分组
        camera_groups = defaultdict(list)
        for cam in viewpoint_stack:
            cam_id = cam.camera_idx if hasattr(cam, 'camera_idx') else cam.colmap_id
            camera_groups[cam_id].append(cam)
        
        # 对每组按时间排序
        for cam_id in camera_groups:
            camera_groups[cam_id].sort(key=lambda x: getattr(x, 'time', 0))
        
        # 找到所有可用的连续帧对
        consecutive_pairs_pool = []
        for cam_id, cam_list in camera_groups.items():
            if len(cam_list) >= 2:
                for i in range(len(cam_list) - 1):
                    cam_t = cam_list[i]
                    cam_t1 = cam_list[i + 1]
                    if hasattr(cam_t, 'time') and hasattr(cam_t1, 'time'):
                        time_diff = abs(cam_t1.time - cam_t.time)
                        if time_diff <= 0.02:
                            consecutive_pairs_pool.append((cam_t, cam_t1))
        
        # 采样连续帧对并从池中移除
        num_pairs_needed = num_flow_samples // 2
        sampled_count = 0
        while sampled_count < num_pairs_needed and len(consecutive_pairs_pool) > 0:
            pair_idx = random.randint(0, len(consecutive_pairs_pool) - 1)
            cam_t, cam_t1 = consecutive_pairs_pool.pop(pair_idx)
            
            # 从池中移除这两个相机
            if cam_t in viewpoint_stack:
                viewpoint_stack.remove(cam_t)
            if cam_t1 in viewpoint_stack:
                viewpoint_stack.remove(cam_t1)
            
            viewpoint_cams.extend([cam_t, cam_t1])
            sampled_count += 1
            
            # 更新连续帧对池（移除包含已采样相机的其他帧对）
            consecutive_pairs_pool = [
                (ct, ct1) for ct, ct1 in consecutive_pairs_pool 
                if ct not in viewpoint_cams and ct1 not in viewpoint_cams
            ]
        
        # 如果没采够，补充到随机采样数
        if len(viewpoint_cams) < num_flow_samples:
            num_random_samples += num_flow_samples - len(viewpoint_cams)
    
    # 2. 随机采样剩余样本
    while len(viewpoint_cams) < batch_size and len(viewpoint_stack) > 0:
        idx = random.randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(idx)
        viewpoint_cams.append(viewpoint_cam)
    
    # 如果池空了，重新填充
    if len(viewpoint_stack) == 0:
        viewpoint_stack.extend(temp_list.copy())
    
    return viewpoint_cams
