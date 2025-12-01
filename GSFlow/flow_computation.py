"""
Unified Optical Flow Computation Module
整合GMFlow和GSFlow的光流计算和Loss计算模块

This module provides:
1. GMFlow ground truth generation
2. GSFlow prediction computation
3. Flow loss calculation
4. Flow visualization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2


class FlowComputation:
    """
    统一的光流计算类，整合GMFlow GT生成和GSFlow预测
    """
    
    def __init__(self, gmflow_model=None, device='cuda'):
        """
        初始化光流计算模块
        
        Args:
            gmflow_model: 预训练的GMFlow模型(可选，用于生成GT)
            device: 计算设备
        """
        self.device = device
        self.gmflow_model = gmflow_model
        
        if gmflow_model is not None:
            self.gmflow_model.eval()
            self.gmflow_model.to(device)
    
    def load_gmflow(self, checkpoint_path):
        """
        加载GMFlow模型
        
        Args:
            checkpoint_path: GMFlow checkpoint路径
        """
        import sys
        import os
        
        # 添加项目根目录到Python路径，确保可以正确导入gmflow包
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 使用绝对导入而不是相对导入
        from gmflow.gmflow import GMFlow
        
        # 创建配置
        config = {
            'num_scales': 1,
            'upsample_factor': 8,
            'feature_channels': 128,
            'attention_type': 'swin',
            'num_transformer_layers': 6,
            'ffn_dim_expansion': 4,
            'num_head': 1,
        }
        
        self.gmflow_model = GMFlow(**config)
        
        # 加载权重
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model' in checkpoint:
                self.gmflow_model.load_state_dict(checkpoint['model'])
            else:
                self.gmflow_model.load_state_dict(checkpoint)
            print(f"✅ GMFlow模型已加载: {checkpoint_path}")
        else:
            print(f"⚠️ 警告: GMFlow checkpoint未找到: {checkpoint_path}")
        
        self.gmflow_model.eval()
        self.gmflow_model.to(self.device)
    
    @torch.no_grad()
    def compute_gmflow_gt(self, image_t, image_t1, normalize=True):
        """
        使用GMFlow计算光流ground truth
        
        Args:
            image_t: 时刻t的图像 [3, H, W] 或 [B, 3, H, W]
            image_t1: 时刻t+1的图像 [3, H, W] 或 [B, 3, H, W]
            normalize: 是否对图像进行归一化
        
        Returns:
            flow_gt: 光流ground truth [2, H, W] 或 [B, 2, H, W]
        """
        if self.gmflow_model is None:
            raise RuntimeError("GMFlow模型未加载，请先调用load_gmflow()或在初始化时提供模型")
        
        # 确保是batch格式
        if image_t.dim() == 3:
            image_t = image_t.unsqueeze(0)
        if image_t1.dim() == 3:
            image_t1 = image_t1.unsqueeze(0)
        
        # 移到正确的设备
        image_t = image_t.to(self.device)
        image_t1 = image_t1.to(self.device)
        
        # GMFlow前向传播
        with torch.no_grad():
            flow_predictions = self.gmflow_model(
                image_t, 
                image_t1,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
            )
            
            # 取最后一个预测(最精细的)
            flow_gt = flow_predictions[-1]  # [B, 2, H, W]
        
        # 如果输入是单张图像，返回单张
        if flow_gt.shape[0] == 1:
            flow_gt = flow_gt.squeeze(0)  # [2, H, W]
        
        return flow_gt
    
    def compute_gsflow_prediction(self, render_pkg_t, render_pkg_t1):
        """
        使用GSFlow从渲染结果计算光流预测
        
        Args:
            render_pkg_t: 时刻t的渲染包(包含gs_per_pixel, conic等数据)
            render_pkg_t1: 时刻t+1的渲染包
        
        Returns:
            flow_pred: 预测的光流 [2, H, W]
        """
        # 检查必要的数据
        required_keys = ["gs_per_pixel", "weight_per_gs_pixel", "conic_2D", 
                        "conic_2D_inv", "proj_2D", "x_mu"]
        
        for key in required_keys:
            if key not in render_pkg_t:
                raise ValueError(f"render_pkg_t缺少必要数据: {key}")
            if key in ["conic_2D", "proj_2D"] and key not in render_pkg_t1:
                raise ValueError(f"render_pkg_t1缺少必要数据: {key}")
        
        # 调用优化的GSFlow计算
        flow_pred = self._calculate_gs_flow_optimized(
            render_pkg_t["gs_per_pixel"],
            render_pkg_t["weight_per_gs_pixel"],
            render_pkg_t1["conic_2D"],
            render_pkg_t["conic_2D_inv"],
            render_pkg_t["proj_2D"],
            render_pkg_t1["proj_2D"],
            render_pkg_t["x_mu"]
        )
        
        return flow_pred
    
    def _calculate_gs_flow_optimized(self, gs_per_pixel, weight_per_gs_pixel, 
                                     next_conic_2D, conic_2D_inv, proj_2D, 
                                     next_proj_2D, x_mu):
        """
        优化的Gaussian流计算(内部方法)
        支持多种输入格式: [2, H, W] 或 [K, 2, H, W]
        """
        conic_2D_inv = conic_2D_inv.detach()
        gs_per_pixel = gs_per_pixel.long()
        K = gs_per_pixel.shape[0]
        
        # 计算conic乘积矩阵
        conv_conv = torch.zeros([conic_2D_inv.shape[0], 2, 2], 
                               device=conic_2D_inv.device, 
                               dtype=conic_2D_inv.dtype)
        
        conv_conv[:, 0, 0] = next_conic_2D[:, 0] * conic_2D_inv[:, 0] + \
                             next_conic_2D[:, 1] * conic_2D_inv[:, 1]
        conv_conv[:, 0, 1] = next_conic_2D[:, 0] * conic_2D_inv[:, 1] + \
                             next_conic_2D[:, 1] * conic_2D_inv[:, 2]
        conv_conv[:, 1, 0] = next_conic_2D[:, 1] * conic_2D_inv[:, 0] + \
                             next_conic_2D[:, 2] * conic_2D_inv[:, 1]
        conv_conv[:, 1, 1] = next_conic_2D[:, 1] * conic_2D_inv[:, 1] + \
                             next_conic_2D[:, 2] * conic_2D_inv[:, 2]
        
        # 处理x_mu的不同形状
        x_mu_ndim = x_mu.ndim
        
        if x_mu_ndim == 4 and x_mu.shape[0] == K:
            x_mu_reshaped = x_mu.permute(0, 2, 3, 1).detach()
            use_broadcasting = False
        elif x_mu_ndim == 3 and x_mu.shape[0] == 2:
            x_mu_base = x_mu.permute(1, 2, 0).detach()
            x_mu_reshaped = x_mu_base.unsqueeze(0)
            use_broadcasting = True
        else:
            raise ValueError(f"不支持的x_mu形状: {x_mu.shape}")
        
        # 索引和应用变换
        conv_indexed = conv_conv[gs_per_pixel]
        
        if use_broadcasting:
            conv_multi = torch.matmul(
                conv_indexed,
                x_mu_reshaped.unsqueeze(-1)
            ).squeeze(-1)
            
            flow_per_pixel = (
                conv_multi + 
                next_proj_2D[gs_per_pixel] - 
                proj_2D[gs_per_pixel].detach() - 
                x_mu_reshaped
            )
        else:
            conv_multi = torch.matmul(
                conv_indexed,
                x_mu_reshaped.unsqueeze(-1)
            ).squeeze(-1)
            
            flow_per_pixel = (
                conv_multi + 
                next_proj_2D[gs_per_pixel] - 
                proj_2D[gs_per_pixel].detach() - 
                x_mu_reshaped
            )
        
        # 加权平均
        weight_normalized = weight_per_gs_pixel / \
                           (weight_per_gs_pixel.sum(dim=0, keepdim=True) + 1e-7)
        
        flow_gs = torch.einsum(
            "khw,khwc->chw",
            weight_normalized.detach(), 
            flow_per_pixel
        )
        
        return flow_gs
    
    def compute_flow_loss(self, flow_pred, flow_gt, height, width, 
                         loss_type='l1', normalize=True, mask=None):
        """
        计算光流Loss
        
        Args:
            flow_pred: 预测的光流 [2, H, W]
            flow_gt: Ground truth光流 [2, H, W]
            height: 图像高度
            width: 图像宽度
            loss_type: Loss类型 ('l1', 'l2', 'smooth_l1')
            normalize: 是否归一化到[-1, 1]
            mask: 可选的mask [1, H, W] 或 [H, W]
        
        Returns:
            loss: 标量tensor
        """
        if normalize:
            flow_pred = flow_pred.clone()
            flow_gt = flow_gt.clone()
            
            flow_pred[0] /= height
            flow_pred[1] /= width
            flow_pred = flow_pred.clamp(-1, 1)
            
            flow_gt[0] /= height
            flow_gt[1] /= width
            flow_gt = flow_gt.clamp(-1, 1)
        
        # 计算差异
        if loss_type == 'l1':
            diff = torch.abs(flow_pred - flow_gt)
        elif loss_type == 'l2':
            diff = (flow_pred - flow_gt) ** 2
        elif loss_type == 'smooth_l1':
            diff = F.smooth_l1_loss(flow_pred, flow_gt, reduction='none')
        else:
            raise ValueError(f"不支持的loss类型: {loss_type}")
        
        # 应用mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            diff = diff * mask
            loss = diff.sum() / (mask.sum() * 2 + 1e-7)
        else:
            loss = diff.mean()
        
        return loss
    
    def compute_flow_loss_with_gmflow(self, image_t, image_t1, 
                                     render_pkg_t, render_pkg_t1,
                                     lambda_flow=1.0, loss_type='l1'):
        """
        完整的光流Loss计算流程
        1. 用GMFlow计算GT
        2. 用GSFlow计算预测
        3. 计算Loss
        
        Args:
            image_t: 时刻t的图像 [3, H, W]
            image_t1: 时刻t+1的图像 [3, H, W]
            render_pkg_t: 时刻t的渲染包
            render_pkg_t1: 时刻t+1的渲染包
            lambda_flow: 光流Loss权重
            loss_type: Loss类型
        
        Returns:
            dict: {
                'loss': 总loss,
                'flow_pred': 预测的光流,
                'flow_gt': Ground truth光流
            }
        """
        H, W = image_t.shape[-2:]
        
        # 1. 计算GMFlow GT
        flow_gt = self.compute_gmflow_gt(image_t, image_t1)
        
        # 2. 计算GSFlow预测
        flow_pred = self.compute_gsflow_prediction(render_pkg_t, render_pkg_t1)
        
        # 3. 计算Loss
        loss = self.compute_flow_loss(flow_pred, flow_gt, H, W, loss_type=loss_type)
        
        return {
            'loss': lambda_flow * loss,
            'flow_pred': flow_pred,
            'flow_gt': flow_gt,
            'raw_loss': loss
        }
    
    def visualize_flow(self, flow, flow_vis_dir=None, image=None, vis_type='color', 
                      clip_percentile=95, step=16):
        """
        可视化光流
        
        Args:
            flow: 光流 [2, H, W]
            flow_vis_dir: 保存路径(可选)
            image: 背景图像(可选) [3, H, W]
            vis_type: 可视化类型 ('color', 'magnitude', 'arrows')
            clip_percentile: 百分位裁剪
            step: 箭头间距(仅对arrows类型有效)
        
        Returns:
            vis_img: 可视化结果 [H, W, 3] numpy array
        """
        # 转换为numpy
        if isinstance(flow, torch.Tensor):
            if flow.shape[0] == 2:
                flow_np = flow.detach().permute(1, 2, 0).cpu().numpy()
            else:
                flow_np = flow.detach().cpu().numpy()
        else:
            flow_np = flow
        
        if vis_type == 'color':
            vis_img = self._flow_to_color(flow_np, clip_percentile=clip_percentile)
            
            # 如果有背景图像，并排显示
            if image is not None:
                if isinstance(image, torch.Tensor):
                    image_np = (image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                else:
                    image_np = image
                vis_img = np.concatenate([image_np, vis_img], axis=1)
                
        elif vis_type == 'magnitude':
            magnitude = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)
            if magnitude.size > 0:
                max_val = np.percentile(magnitude, clip_percentile)
                magnitude = np.clip(magnitude, 0, max_val) / (max_val + 1e-7)
            magnitude_norm = (magnitude * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
        elif vis_type == 'arrows':
            if image is None:
                H, W = flow_np.shape[:2]
                vis_img = np.ones((H, W, 3), dtype=np.uint8) * 255
            else:
                if isinstance(image, torch.Tensor):
                    vis_img = (image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                else:
                    vis_img = image.copy()
            
            vis_img = self._draw_flow_arrows(vis_img, flow_np, step=step)
        else:
            raise ValueError(f"不支持的可视化类型: {vis_type}")
        
        # 保存
        if flow_vis_dir is not None:
            os.makedirs(flow_vis_dir, exist_ok=True)
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(flow_vis_dir, "flow_vis.png"), vis_img_bgr)
        
        return vis_img
    
    def _flow_to_color(self, flow, clip_percentile=95):
        """将光流转换为HSV彩色图"""
        h, w = flow.shape[:2]
        dx = flow[..., 0]
        dy = flow[..., 1]
        
        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        if magnitude.size > 0:
            max_flow = np.percentile(magnitude, clip_percentile)
            max_flow = max(max_flow, 1e-7)
        else:
            max_flow = 1e-7
        
        scaled_magnitude = magnitude / max_flow
        
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
        hsv[..., 1] = 255
        value = np.clip(np.power(scaled_magnitude, 0.5) * 255, 0, 255).astype(np.uint8)
        hsv[..., 2] = value
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
    
    def _draw_flow_arrows(self, image, flow, step=16):
        """在图像上绘制光流箭头"""
        h, w = flow.shape[:2]
        result = image.copy()
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        ref_magnitude = np.percentile(magnitude, 95) if magnitude.size > 0 else 10
        scale = min(step / (ref_magnitude + 1e-7), 5.0)
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                mag = np.sqrt(fx**2 + fy**2)
                if mag > 0.5:  # 只绘制有意义的光流
                    end_x = int(x + fx * scale)
                    end_y = int(y + fy * scale)
                    cv2.arrowedLine(result, (x, y), (end_x, end_y), 
                                   (0, 255, 0), 1, tipLength=0.3)
        
        return result


# 便捷函数
def create_flow_computer(gmflow_checkpoint=None, device='cuda'):
    """
    创建FlowComputation实例的便捷函数
    
    Args:
        gmflow_checkpoint: GMFlow checkpoint路径
        device: 计算设备
    
    Returns:
        FlowComputation实例
    """
    flow_computer = FlowComputation(device=device)
    
    if gmflow_checkpoint is not None:
        flow_computer.load_gmflow(gmflow_checkpoint)
    
    return flow_computer
