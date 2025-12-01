"""
Flow Visualization Manager
独立的光流可视化管理模块，从训练循环中分离可视化逻辑

这个模块负责：
1. 决定何时进行可视化
2. 执行可视化并保存
3. 管理可视化输出目录
"""

import torch
import os
import numpy as np
from pathlib import Path


class FlowVisualizationManager:
    """
    光流可视化管理器
    """
    
    def __init__(self, flow_computer, output_dir, vis_interval=500, 
                 enable_visualization=True):
        """
        初始化可视化管理器
        
        Args:
            flow_computer: FlowComputation实例
            output_dir: 输出根目录
            vis_interval: 可视化间隔（迭代数）
            enable_visualization: 是否启用可视化
        """
        self.flow_computer = flow_computer
        self.vis_interval = vis_interval
        self.enable_visualization = enable_visualization
        
        # 创建可视化输出目录
        if enable_visualization:
            self.flow_vis_dir = Path(output_dir) / "flow_vis"
            self.flow_vis_dir.mkdir(parents=True, exist_ok=True)
            print(f"光流可视化将保存到: {self.flow_vis_dir}")
        else:
            self.flow_vis_dir = None
    
    def should_visualize(self, iteration):
        """
        判断当前迭代是否应该进行可视化
        
        Args:
            iteration: 当前迭代数
        
        Returns:
            bool: 是否应该可视化
        """
        if not self.enable_visualization:
            return False
        
        return iteration % self.vis_interval == 0
    
    def visualize_and_save(self, iteration, flow_pred, flow_gt=None, 
                          image_t=None, image_t1=None):
        """
        可视化并保存光流结果
        
        Args:
            iteration: 当前迭代数
            flow_pred: 预测的光流 [2, H, W]
            flow_gt: Ground truth光流(可选) [2, H, W]
            image_t: 时刻t的图像(可选) [3, H, W]
            image_t1: 时刻t+1的图像(可选) [3, H, W]
        """
        if not self.enable_visualization or self.flow_vis_dir is None:
            return
        
        # 创建迭代专用子目录
        iter_dir = self.flow_vis_dir / f"iter_{iteration:06d}"
        iter_dir.mkdir(exist_ok=True)
        
        # 1. 可视化预测光流（色彩编码）
        flow_pred_vis = self.flow_computer.visualize_flow(
            flow_pred,
            flow_vis_dir=None,  # 不直接保存，我们手动保存
            vis_type='color'
        )
        self._save_image(flow_pred_vis, iter_dir / "flow_pred_color.png")
        
        # 2. 可视化预测光流（幅度热力图）
        flow_pred_mag = self.flow_computer.visualize_flow(
            flow_pred,
            flow_vis_dir=None,
            vis_type='magnitude'
        )
        self._save_image(flow_pred_mag, iter_dir / "flow_pred_magnitude.png")
        
        # 3. 如果有GT，可视化GT
        if flow_gt is not None:
            flow_gt_vis = self.flow_computer.visualize_flow(
                flow_gt,
                flow_vis_dir=None,
                vis_type='color'
            )
            self._save_image(flow_gt_vis, iter_dir / "flow_gt_color.png")
            
            # 计算并可视化误差
            error = self._compute_flow_error(flow_pred, flow_gt)
            self._save_image(error, iter_dir / "flow_error.png")
        
        # 4. 如果有图像，绘制箭头
        if image_t is not None:
            flow_arrows = self.flow_computer.visualize_flow(
                flow_pred,
                flow_vis_dir=None,
                image=image_t,
                vis_type='arrows',
                step=20
            )
            self._save_image(flow_arrows, iter_dir / "flow_arrows.png")
        
        # 5. 保存原始图像（用于对比）
        if image_t is not None:
            self._save_tensor_image(image_t, iter_dir / "image_t.png")
        if image_t1 is not None:
            self._save_tensor_image(image_t1, iter_dir / "image_t1.png")
        
        # 6. 保存光流统计信息
        self._save_flow_statistics(flow_pred, flow_gt, iter_dir / "statistics.txt", iteration)
        
        print(f"✅ [Iter {iteration}] 光流可视化已保存到: {iter_dir}")
    
    def _compute_flow_error(self, flow_pred, flow_gt):
        """计算光流误差的可视化"""
        import cv2
        
        if isinstance(flow_pred, torch.Tensor):
            flow_pred_np = flow_pred.detach().permute(1, 2, 0).cpu().numpy()
        else:
            flow_pred_np = flow_pred
        
        if isinstance(flow_gt, torch.Tensor):
            flow_gt_np = flow_gt.detach().permute(1, 2, 0).cpu().numpy()
        else:
            flow_gt_np = flow_gt
        
        # 计算端点误差 (EPE)
        epe = np.sqrt(np.sum((flow_pred_np - flow_gt_np)**2, axis=-1))
        
        # 归一化并着色
        epe_norm = np.clip(epe / (np.percentile(epe, 95) + 1e-7), 0, 1)
        epe_vis = (epe_norm * 255).astype(np.uint8)
        epe_colored = cv2.applyColorMap(epe_vis, cv2.COLORMAP_JET)
        epe_rgb = cv2.cvtColor(epe_colored, cv2.COLOR_BGR2RGB)
        
        return epe_rgb
    
    def _save_image(self, img, path):
        """保存numpy图像"""
        import cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img_bgr)
    
    def _save_tensor_image(self, tensor_img, path):
        """保存torch tensor图像"""
        import cv2
        if isinstance(tensor_img, torch.Tensor):
            img_np = (tensor_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = (tensor_img * 255).astype(np.uint8)
        
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img_bgr)
    
    def _save_flow_statistics(self, flow_pred, flow_gt, path, iteration):
        """保存光流统计信息到文本文件"""
        if isinstance(flow_pred, torch.Tensor):
            flow_pred_np = flow_pred.detach().cpu().numpy()
        else:
            flow_pred_np = flow_pred
        
        # 计算预测光流统计
        dx_pred = flow_pred_np[0] if flow_pred_np.shape[0] == 2 else flow_pred_np[..., 0]
        dy_pred = flow_pred_np[1] if flow_pred_np.shape[0] == 2 else flow_pred_np[..., 1]
        mag_pred = np.sqrt(dx_pred**2 + dy_pred**2)
        
        stats_lines = [
            f"Iteration: {iteration}",
            f"\n{'='*60}",
            f"\nPredicted Flow Statistics:",
            f"  Magnitude: mean={mag_pred.mean():.4f}, max={mag_pred.max():.4f}, "
            f"min={mag_pred.min():.4f}, std={mag_pred.std():.4f}",
            f"  DX: mean={dx_pred.mean():.4f}, max={dx_pred.max():.4f}, min={dx_pred.min():.4f}",
            f"  DY: mean={dy_pred.mean():.4f}, max={dy_pred.max():.4f}, min={dy_pred.min():.4f}",
        ]
        
        # 如果有GT，计算误差统计
        if flow_gt is not None:
            if isinstance(flow_gt, torch.Tensor):
                flow_gt_np = flow_gt.detach().cpu().numpy()
            else:
                flow_gt_np = flow_gt
            
            dx_gt = flow_gt_np[0] if flow_gt_np.shape[0] == 2 else flow_gt_np[..., 0]
            dy_gt = flow_gt_np[1] if flow_gt_np.shape[0] == 2 else flow_gt_np[..., 1]
            
            # 端点误差 (EPE)
            epe = np.sqrt((dx_pred - dx_gt)**2 + (dy_pred - dy_gt)**2)
            
            stats_lines.extend([
                f"\n{'='*60}",
                f"\nGround Truth Flow Statistics:",
                f"  Magnitude: mean={np.sqrt(dx_gt**2 + dy_gt**2).mean():.4f}",
                f"\n{'='*60}",
                f"\nError Statistics:",
                f"  EPE (End-Point Error): mean={epe.mean():.4f}, max={epe.max():.4f}, "
                f"min={epe.min():.4f}, std={epe.std():.4f}",
                f"  Pixels with EPE > 3: {(epe > 3).sum()} ({(epe > 3).mean()*100:.2f}%)",
            ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(stats_lines))
    
    def create_summary_video(self, output_path="flow_evolution.mp4", fps=10):
        """
        从保存的可视化创建演化视频（可选功能）
        
        Args:
            output_path: 输出视频路径
            fps: 帧率
        """
        if not self.enable_visualization or self.flow_vis_dir is None:
            print("⚠️ 可视化未启用，无法创建视频")
            return
        
        try:
            import subprocess
            
            # 使用ffmpeg创建视频
            pattern = str(self.flow_vis_dir / "iter_*/flow_pred_color.png")
            full_output = str(self.flow_vis_dir / output_path)
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-pattern_type', 'glob',
                '-i', pattern,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                full_output
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ 光流演化视频已创建: {full_output}")
            
        except Exception as e:
            print(f"⚠️ 创建视频失败: {e}")
            print("   提示: 确保安装了ffmpeg")


def create_visualization_manager(flow_computer, output_dir, vis_interval=500, 
                                enable=True):
    """
    创建可视化管理器的便捷函数
    
    Args:
        flow_computer: FlowComputation实例
        output_dir: 输出目录
        vis_interval: 可视化间隔
        enable: 是否启用
    
    Returns:
        FlowVisualizationManager实例
    """
    return FlowVisualizationManager(
        flow_computer=flow_computer,
        output_dir=output_dir,
        vis_interval=vis_interval,
        enable_visualization=enable
    )
