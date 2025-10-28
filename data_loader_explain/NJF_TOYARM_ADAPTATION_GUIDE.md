# NJF ToyArm数据集适配4DGaussians指南

## 数据格式对比分析

### NJF ToyArm 数据格式 (从transforms.json片段分析)
```json
{
    "cameras": [
        {
            "transform_matrix": [[...], [...], [...], [...]],  // 4x4相机外参矩阵
            "fl_x": 606.572265625,  // 焦距x
            "fl_y": 605.9955444335938,  // 焦距y
            "cx": 327.9910583496094,  // 主点x
            "cy": 239.932373046875,  // 主点y
            "h": 480,  // 图像高度
            "w": 640,  // 图像宽度
            "camera_model": "OPENCV"
        },
        // ... 多个相机
    ],
    "frames": [
        {
            "file_path": "view_X/rgb/XXXXX_XXXXX.png",
            "depth_file_path": "view_X/depth/XXXXX_XXXXX.png",
            "time": 0.0~1.0,  // 归一化时间戳
            "sample_idx": X,  // 样本索引
            "camera_idx": X,  // 相机索引
            "joint_pos": [...]  // 机械臂关节位置
        },
        // ... 大量帧数据
    ]
}
```

**特点：**
1. **相机与帧分离**: 相机内外参在顶层`cameras`数组，帧数据通过`camera_idx`引用
2. **多视角多时间步**: 每个时间步有多个相机视角(view_0~view_11)
3. **完整内参**: 提供了`fl_x`, `fl_y`, `cx`, `cy`等完整内参
4. **额外数据**: 包含`depth_file_path`和`joint_pos`等额外信息

### 4DGaussians Blender格式 (当前期望格式)
```json
{
    "camera_angle_x": 0.8575560450553894,  // 或 "fl_x", "fl_y"
    "frames": [
        {
            "file_path": "./train/r_0",
            "transform_matrix": [[...], [...], [...], [...]],  // 4x4相机外参矩阵
            "time": 0.0~1.0,  // 归一化时间戳
            "u": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // 关节位置差值（新增）
            "joint_pos": [-6.3, -9.82, -8.65, -11.0, 1.03, 6.01]  // 原始关节位置（调试用）
        },
        // ... 所有帧数据
    ]
}
```

**特点：**
1. **相机与帧耦合**: 每个frame包含完整的相机参数
2. **简化内参**: 通常只有`camera_angle_x`或`fl_x/fl_y`
3. **标准NeRF格式**: 遵循原始NeRF论文的数据格式
4. **新增u字段**: 关节位置差值，用于条件化训练 (u = current_joint_pos - previous_joint_pos)

## 适配方案

### 方案一：生成新的transforms.json文件（推荐）

#### 优点
- 无需修改4DGaussians代码
- 保持4DGaussians原有的稳定性
- 便于数据管理和调试

#### 实现步骤
使用提供的转换脚本 `convert_njf_to_4dgs.py` (见下方)

#### 转换脚本说明

转换脚本会完成以下任务：
1. 读取NJF格式的transforms.json
2. 将相机内外参合并到每一帧
3. **计算关节位置差值u**（每个相机独立计算：u = current - previous）
4. 生成标准的`transforms_train.json`和`transforms_test.json`（包含u字段）
5. 可选：生成点云初始化文件

**关于u（关节位置差值）**：
- 自动为每个frame计算 u = current_joint_pos - previous_joint_pos
- 每个相机视角独立计算（12个相机各自跟踪）
- 第一帧 u = [0, 0, 0, 0, 0, 0]
- 可用于条件化4DGaussians训练（控制信号）
- 详见：`关节位置差值u的使用说明.md`

### 方案二：修改4DGaussians代码以支持NJF格式

#### 需要修改的文件

1. **scene/dataset_readers.py** - 核心修改
   - 新增 `readNJFTransforms()` 函数
   - 修改 `sceneLoadTypeCallbacks` 字典

2. **train.py** 或相关配置文件
   - 添加新的数据集类型选项

#### 具体修改内容（见下方代码实现）

## 使用建议

### 推荐工作流程

1. **使用方案一（转换脚本）进行初步测试**
   ```bash
   python convert_njf_to_4dgs.py --input /path/to/njf/transforms.json --output /path/to/4dgs/data
   ```

2. **验证转换结果**
   - 检查生成的transforms_train.json和transforms_test.json
   - 确认图像路径正确
   - 验证相机参数合理性

3. **运行4DGaussians训练**
   ```bash
   python train.py -s /path/to/4dgs/data -m output/toyarm --eval
   ```

4. **如果需要频繁使用NJF数据，再考虑方案二（代码修改）**

## 关键差异处理

### 1. 相机坐标系统
- **NJF**: 使用OpenCV坐标系（可能需要转换）
- **4DGaussians**: 期望OpenGL/Blender坐标系

转换公式在脚本中已实现：
```python
# 如果需要坐标系转换
c2w[:3, 1:3] *= -1  # Y和Z轴翻转
```

### 2. 时间归一化
- 两者都使用0~1的归一化时间，无需额外处理

### 3. 训练/测试集划分
- NJF没有明确划分，需要手动split
- 转换脚本提供了自动划分功能（默认9:1）

### 4. 点云初始化
- 如果没有COLMAP数据，4DGaussians会生成随机点云
- 可以利用NJF的depth数据生成更好的初始点云（可选功能）

## 测试检查清单

- [ ] 转换后的JSON格式正确
- [ ] 图像路径可以正确访问
- [ ] 相机内参数值合理（焦距、主点）
- [ ] 相机外参导致的场景范围合理
- [ ] 训练/测试集划分合理
- [ ] 时间戳正确且递增
- [ ] （可选）初始点云质量良好

## 注意事项

1. **图像路径**: NJF使用`view_X/rgb/`结构，确保4DGaussians能正确找到图像
2. **内存占用**: 多视角数据会显著增加内存需求
3. **训练时间**: 帧数较多时训练时间会延长
4. **相机标定**: 确保NJF的相机标定准确，否则会影响重建质量

## 故障排除

### 问题：找不到图像文件
- 检查转换脚本中的`data_root`路径设置
- 确认相对路径正确

### 问题：相机视角异常
- 检查坐标系转换是否正确
- 验证transform_matrix的正负号

### 问题：训练不收敛
- 减少初始点云数量（num_pts参数）
- 调整学习率
- 检查时间归一化是否正确

## 下一步

根据你的需求，请查看：
1. `convert_njf_to_4dgs.py` - 数据转换脚本
2. `njf_toyarm_dataloader.py` - 4DGaussians的自定义数据加载器（方案二）

---
*本指南基于4DGaussians v1.0和Neural-Jacobian-Field的ToyArm数据集*
