# 方案二：修改4DGaussians以直接支持NJF格式

本文件说明如何修改4DGaussians代码以直接加载NJF ToyArm数据集。

## 修改步骤

### 1. 复制NJF加载器到项目

已创建的文件：
- `scene/njf_toyarm_loader.py` - NJF数据加载器（包含u计算功能）

**新增功能**：数据加载器会自动计算关节位置差值u
- 使用 `CameraInfoWithU` 类（扩展了标准CameraInfo）
- 包含 `u` 和 `joint_pos` 字段
- 运行时动态计算，逻辑与转换脚本一致

### 2. 修改 `scene/dataset_readers.py`

在文件末尾的 `sceneLoadTypeCallbacks` 字典中添加NJF支持：

```python
# 在文件开头添加导入
from scene.njf_toyarm_loader import readNJFSceneInfo

# 在文件末尾修改sceneLoadTypeCallbacks字典
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos,
    "NJF": readNJFSceneInfo,  # 新增这一行
}
```

**具体修改位置**：`scene/dataset_readers.py` 第679行附近

### 3. 修改 `train.py` 或 `arguments/__init__.py`

添加数据集类型选项（如果需要命令行参数）：

在 `arguments/__init__.py` 中找到 `ModelParams` 类，添加：

```python
class ModelParams:
    # ... 其他参数 ...
    
    def __init__(self, parser, sentinel=False):
        # ... 其他初始化代码 ...
        
        # 在source_path参数附近添加
        self.parser.add_argument('--source_path', '-s', type=str, default="")
        self.parser.add_argument('--dataset_type', type=str, 
                                default='Colmap',
                                choices=['Colmap', 'Blender', 'dynerf', 'nerfies', 
                                        'PanopticSports', 'MultipleView', 'NJF'],
                                help='Type of dataset to load')
```

或者，如果4DGaussians使用配置文件系统，在相应配置文件中添加 `dataset_type` 选项。

### 4. 使用修改后的代码

训练命令：

```bash
python train.py -s /path/to/njf/toyarm/data --dataset_type NJF -m output/toyarm --eval
```

## 详细代码修改示例

### scene/dataset_readers.py 修改示例

在文件开头添加导入（约第30行）：

```python
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
from scene.njf_toyarm_loader import readNJFSceneInfo  # 新增这一行
```

在文件末尾修改字典（约第679行）：

```python
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos,
    "NJF": readNJFSceneInfo,  # 新增：Neural-Jacobian-Field ToyArm dataset
}
```

### scene/__init__.py 修改（如果需要）

找到加载场景的代码（通常在 `Scene` 类的 `__init__` 方法中），确保它使用了 `sceneLoadTypeCallbacks`：

```python
class Scene:
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        # ... 其他代码 ...
        
        # 确保使用了正确的dataset_type
        dataset_type = getattr(args, 'dataset_type', 'Colmap')
        
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif dataset_type == "NJF":  # 新增这个判断
            print("Using NJF dataset loader")
            scene_info = sceneLoadTypeCallbacks["NJF"](args.source_path, args.white_background, args.eval)
        else:
            # ... 其他判断 ...
```

## 数据准备

### NJF数据集目录结构

确保你的NJF数据集结构如下：

```
toyarm_data/
├── transforms.json          # NJF格式的主配置文件
├── view_0/
│   ├── rgb/
│   │   ├── 01022_00000.png
│   │   ├── 01022_00001.png
│   │   └── ...
│   └── depth/
│       ├── 01022_00000.png
│       └── ...
├── view_1/
│   └── ...
└── ... (view_2 到 view_11)
```

### 可选：预生成点云

为了获得更好的初始化，可以预先生成 `fused.ply`：

```python
python -c "
from scene.njf_toyarm_loader import depth_to_pointcloud
import numpy as np
# 从depth图生成点云的代码
"
```

或使用转换脚本：

```bash
python convert_njf_to_4dgs.py --input /path/to/transforms.json --output /tmp/toyarm_converted
# 然后复制生成的fused.ply到你的NJF数据目录
cp /tmp/toyarm_converted/fused.ply /path/to/njf/toyarm/data/
```

## 测试修改

### 1. 验证数据加载

```python
from scene.njf_toyarm_loader import readNJFSceneInfo

# 测试加载
scene_info = readNJFSceneInfo("/path/to/njf/toyarm/data", white_background=False, eval=True)

print(f"Train cameras: {len(scene_info.train_cameras)}")
print(f"Test cameras: {len(scene_info.test_cameras)}")
print(f"Point cloud: {scene_info.point_cloud.points.shape}")
```

### 2. 运行完整训练

```bash
# 使用NJF数据集训练
python train.py -s /path/to/njf/toyarm/data --dataset_type NJF -m output/toyarm_test --eval --iterations 7000

# 或者如果自动检测起作用
python train.py -s /path/to/njf/toyarm/data -m output/toyarm_test --eval
```

### 3. 渲染测试

```bash
python render.py -m output/toyarm_test
```

## 常见问题

### Q1: 相机视角不对/场景倒置

**原因**：坐标系转换问题

**解决**：在 `scene/njf_toyarm_loader.py` 的 `readNJFTransforms` 函数中，找到以下注释的代码：

```python
# 可能需要坐标系转换（从OpenCV到OpenGL）
# 如果渲染结果不对，可以尝试取消这个注释
# c2w[:3, 1:3] *= -1
```

取消注释试试：

```python
# 坐标系转换（从OpenCV到OpenGL）
c2w[:3, 1:3] *= -1
```

### Q2: 训练不收敛

**可能原因**：
1. 初始点云不合理
2. 相机参数错误
3. 时间归一化问题

**解决步骤**：
1. 检查生成的fused.ply是否在合理范围内
2. 可视化相机位置（使用4DGaussians自带工具）
3. 验证时间戳在0-1范围内

### Q3: 找不到图像文件

**原因**：路径问题

**解决**：确保transforms.json中的file_path是相对于数据集根目录的正确路径

## 高级功能（TODO）

### 使用Depth数据初始化点云

修改 `scene/njf_toyarm_loader.py` 中的 `readNJFSceneInfoWithDepth` 函数，实现：

1. 读取所有depth图
2. 使用 `depth_to_pointcloud` 函数转换为3D点
3. 合并所有视角的点云
4. 下采样并保存为fused.ply

### 使用Joint Position信息

如果需要利用机械臂的关节位置信息（例如用于条件生成），可以：

1. 在 `CameraInfo` 中添加 `joint_pos` 字段
2. 在训练循环中使用这些信息

## 性能优化建议

1. **减少加载的视角数**：如果12个视角太多，可以在loader中添加筛选逻辑
2. **图像下采样**：修改 `PILtoTorch(image, (800, 800))` 中的尺寸
3. **点云下采样**：减少初始点云数量

## 回退到方案一

如果修改代码遇到问题，可以随时回退到方案一（使用转换脚本）：

```bash
# 使用转换脚本
python convert_njf_to_4dgs.py --input /path/to/njf/transforms.json --output /path/to/converted_data

# 使用标准Blender格式训练
python train.py -s /path/to/converted_data -m output/toyarm --eval
```

---
*这些修改基于4DGaussians的标准架构，如果你的版本有所不同，请根据实际情况调整*
