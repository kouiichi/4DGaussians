# NJF ToyArm 数据集适配 4DGaussians

本仓库包含将 Neural-Jacobian-Field (NJF) ToyArm 数据集适配到 4DGaussians 进行训练的完整解决方案。

## 📁 文件结构

```
4DGaussians/
├── convert_njf_to_4dgs.py              # 数据转换脚本（方案一）
├── scene/
│   └── njf_toyarm_loader.py            # NJF数据加载器（方案二）
├── QUICKSTART.md                       # 快速开始指南 ⭐
├── NJF_TOYARM_ADAPTATION_GUIDE.md     # 详细适配指南
└── INTEGRATION_GUIDE.md                # 代码集成指南（方案二）
```

## 🚀 快速开始

### 推荐方案：数据转换

```bash
# 1. 转换数据格式
python convert_njf_to_4dgs.py \
    --input /path/to/njf/transforms.json \
    --output ./data/toyarm_converted

# 2. 训练模型
python train.py \
    -s ./data/toyarm_converted \
    -m ./output/toyarm \
    --eval

# 3. 渲染结果
python render.py -m ./output/toyarm
```

详细步骤请查看 **[QUICKSTART.md](QUICKSTART.md)** ⭐

## 📚 文档说明

### 1. [QUICKSTART.md](QUICKSTART.md) ⭐ **从这里开始！**
- 5分钟快速开始教程
- 完整命令示例
- 常见问题解答
- 性能优化技巧

### 2. [NJF_TOYARM_ADAPTATION_GUIDE.md](NJF_TOYARM_ADAPTATION_GUIDE.md)
- NJF与4DGaussians数据格式对比
- 两种适配方案详细说明
- 关键差异处理方法
- 测试检查清单

### 3. [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- 方案二（代码修改）的详细实现
- 具体代码修改位置和内容
- 高级功能实现（depth-based初始化）
- 故障排除指南

## 🎯 两种适配方案对比

| 特性 | 方案一：数据转换 | 方案二：代码集成 |
|------|-----------------|-----------------|
| **实现难度** | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| **修改代码** | ❌ 不需要 | ✅ 需要 |
| **稳定性** | ⭐⭐⭐ 高 | ⭐⭐ 中 |
| **灵活性** | ⭐⭐ 中 | ⭐⭐⭐ 高 |
| **推荐场景** | 快速测试、一次性使用 | 频繁使用、自定义需求 |

**建议**：先用方案一测试，确认可行后再考虑方案二。

## 📋 数据格式说明

### NJF ToyArm 格式特点

```json
{
    "cameras": [
        {
            "transform_matrix": [[...]],  // 相机外参
            "fl_x": 606.57,              // 焦距
            "cx": 327.99,                // 主点
            ...
        }
    ],
    "frames": [
        {
            "file_path": "view_0/rgb/01022_00000.png",
            "time": 0.0,
            "camera_idx": 0,
            "joint_pos": [...]
        }
    ]
}
```

**关键特征**：
- 相机与帧分离设计
- 多视角（12个相机）
- 多时间步动态场景
- 包含depth和joint_pos等额外信息

### 转换后的4DGaussians格式

```json
{
    "camera_angle_x": 0.857,
    "frames": [
        {
            "file_path": "view_0/rgb/01022_00000.png",
            "transform_matrix": [[...]],
            "time": 0.0
        }
    ]
}
```

**标准NeRF格式**：相机参数合并到每一帧中。

## 🔧 工具说明

### convert_njf_to_4dgs.py

数据格式转换脚本，功能包括：

- ✅ NJF格式转换为4DGaussians/Blender格式
- ✅ 自动训练/测试集划分
- ✅ 生成初始点云
- ✅ 可选的坐标系转换
- ✅ 可选的图像复制

**使用示例**：

```bash
# 基本用法
python convert_njf_to_4dgs.py \
    --input /path/to/transforms.json \
    --output ./data/output

# 完整参数
python convert_njf_to_4dgs.py \
    --input /path/to/transforms.json \
    --output ./data/output \
    --train_split 0.9 \
    --copy_images \
    --coord_transform opencv_to_opengl \
    --camera_angle_x 0.857
```

查看所有参数：
```bash
python convert_njf_to_4dgs.py --help
```

### scene/njf_toyarm_loader.py

方案二的NJF数据加载器，提供：

- 直接读取NJF格式transforms.json
- 自动处理相机参数合并
- 可选的坐标系转换
- 支持depth-based点云初始化（实验性）

## 🧪 测试流程

### 1. 验证数据转换

```bash
# 转换数据
python convert_njf_to_4dgs.py --input ... --output ./test_output

# 检查输出
ls ./test_output/
# 应该看到：transforms_train.json, transforms_test.json, fused.ply

# 验证JSON格式
python -c "
import json
train = json.load(open('./test_output/transforms_train.json'))
print(f'Train frames: {len(train[\"frames\"])}')
print(f'Camera angle X: {train[\"camera_angle_x\"]}')
"
```

### 2. 快速训练测试

```bash
# 短时间训练（验证流程）
python train.py \
    -s ./test_output \
    -m ./test_model \
    --iterations 1000

# 检查是否有错误
ls ./test_model/
```

### 3. 完整训练

```bash
# 完整训练
python train.py \
    -s ./data/toyarm \
    -m ./output/toyarm_final \
    --eval \
    --iterations 7000
```

## 📊 性能建议

### 硬件要求

- **GPU**: 至少8GB显存（推荐12GB+）
- **内存**: 16GB+
- **存储**: 取决于数据集大小

### 优化技巧

1. **减少训练数据**：
   ```bash
   --train_split 0.5  # 只用50%数据
   ```

2. **降低图像分辨率**：
   修改loader中的 `PILtoTorch(image, (400, 400))` 

3. **减少迭代次数**：
   ```bash
   --iterations 3000  # 从7000降到3000
   ```

## ❓ 常见问题

### Q1: 转换脚本找不到图像

**解决**：
- 确认transforms.json中的路径正确
- 使用 `--copy_images` 参数复制图像

### Q2: 训练时相机视角不对

**解决**：
- 使用 `--coord_transform opencv_to_opengl` 参数
- 检查相机外参矩阵

### Q3: 训练不收敛

**解决**：
- 检查时间归一化（应在0-1范围）
- 验证初始点云位置合理
- 尝试调整学习率

更多问题请查看 [QUICKSTART.md](QUICKSTART.md) 的常见问题部分。

## 🔄 更新日志

### v1.0 (2025-10-28)
- ✅ 初始版本
- ✅ 支持NJF ToyArm数据集
- ✅ 提供两种适配方案
- ✅ 完整文档和示例

## 📖 相关资源

- [4DGaussians](https://github.com/hustvl/4DGaussians) - 原始4DGaussians项目
- [Neural-Jacobian-Field](https://github.com/sizhe-li/neural-jacobian-field) - NJF项目
- [NeRF](https://www.matthewtancik.com/nerf) - NeRF原始论文

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

本适配代码遵循 4DGaussians 的许可证。

---

**开始使用**：查看 [QUICKSTART.md](QUICKSTART.md) 进行快速开始！ 🚀
