# GSFlow Module
# Gaussian Flow calculation for 4D Gaussian Splatting with control vector
# 整合光流计算、可视化和Loss计算的统一模块

from .flow_computation import FlowComputation, create_flow_computer
from .flow_visualization_manager import FlowVisualizationManager, create_visualization_manager
from .camera_sampler import (
    group_cameras_by_view,
    sample_consecutive_frames_same_camera,
    extract_camera_id_from_name,
    get_camera_statistics,
    validate_frame_pair,
    create_camera_sampler
)

__all__ = [
    'FlowComputation',
    'create_flow_computer',
    'FlowVisualizationManager',
    'create_visualization_manager',
    'group_cameras_by_view',
    'sample_consecutive_frames_same_camera',
    'extract_camera_id_from_name',
    'get_camera_statistics',
    'validate_frame_pair',
    'create_camera_sampler'
]
