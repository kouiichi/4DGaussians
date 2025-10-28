# NJF ToyArm 适配 4DGaussians - 操作手册

本文档说明如何使用 Neural-Jacobian-Field (NJF) ToyArm 数据集训练 4DGaussians，提供两种方案及其所需的代码修改和操作步骤。

---

## 📋 核心内容概览

### 两种适配方案

| 方案 | 核心思想 | 代码修改 | 适用场景 |
|-----|---------|---------|---------|
| **方案一：数据转换** | 转换NJF格式为4DGaussians格式 | ❌ 无需修改 | 快速测试、一次性使用 |
| **方案二：代码集成** | 修改4DGaussians直接读取NJF | ✅ 需要修改 | 频繁使用、自定义需求 |

### 新增功能：关节位置差值 u

两种方案都会自动计算关节位置差值 `u`：
- **u = current_joint_pos - previous_joint_pos** (每个相机独立计算)
- **第一帧**: u = [0, 0, 0, 0, 0, 0]
- **用途**: 可作为条件信号用于控制化4DGaussians训练

---

## 🚀 方案一：数据转换（推荐新手）

### 需要的文件

```
4DGaussians/
└── convert_njf_to_4dgs.py    # 数据转换脚本（已提供）
```

### 代码修改

**✅ 无需任何代码修改！**

### 操作步骤

#### 第一步：转换数据

```bash
cd d:\Codee\4DGaussians

python convert_njf_to_4dgs.py \
    --input d:\Codee\neural-jacobian-field\data\toyarm\transforms.json \
    --output d:\Codee\4DGaussians\data\toyarm_converted \
    --train_split 0.9
```

**参数说明**：
- `--input`: NJF的transforms.json路径
- `--output`: 输出目录（新建，不会覆盖原始数据）
- `--train_split`: 训练集比例（默认0.9）

**生成的文件**：
```
data/toyarm_converted/
├── transforms_train.json   # 训练集（含u字段）
├── transforms_test.json    # 测试集（含u字段）
└── fused.ply              # 初始点云
```

**transforms.json 格式示例**：
```json
{
    "camera_angle_x": 0.857,
    "frames": [
        {
            "file_path": "view_0/rgb/00000_00000.png",
            "transform_matrix": [[...], ...],
            "time": 0.0,
            "u": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // 关节位置差值
            "joint_pos": [-6.3, -9.82, -8.65, -11.0, 1.03, 6.01],
            "sample_idx": 0,
            "camera_idx": 0
        }
    ]
}
```

#### 第二步：创建图像符号链接

```powershell
cd d:\Codee\4DGaussians\data\toyarm_converted

# 为每个视角创建符号链接
0..11 | ForEach-Object { 
    cmd /c mklink /J "view_$_" "d:\Codee\neural-jacobian-field\data\toyarm\view_$_" 
}
```

**或者**复制图像（占用更多空间）：
```bash
python convert_njf_to_4dgs.py --input ... --output ... --copy_images
```

#### 第三步：训练

```bash
python train.py \
    -s d:\Codee\4DGaussians\data\toyarm_converted \
    -m d:\Codee\4DGaussians\output\toyarm_exp1 \
    --eval \
    --iterations 7000
```

#### 第四步：渲染

```bash
# 渲染训练集
python render.py -m d:\Codee\4DGaussians\output\toyarm_exp1

# 渲染测试集
python render.py -m d:\Codee\4DGaussians\output\toyarm_exp1 --skip_train
```

### 验证

```bash
# 检查u字段
python -c "import json; data=json.load(open('data/toyarm_converted/transforms_train.json')); print('First frame u:', data['frames'][0]['u'])"

# 检查帧数
python -c "import json; print('Train frames:', len(json.load(open('data/toyarm_converted/transforms_train.json'))['frames']))"
```

---

## ⚙️ 方案二：代码集成（推荐高级用户）

### 需要的文件

```
4DGaussians/
├── scene/
│   ├── njf_toyarm_loader.py       # NJF数据加载器（已提供）
│   ├── dataset_readers.py         # 需要修改
│   └── __init__.py                # 可能需要修改
└── arguments/__init__.py          # 可能需要修改
```

### 代码修改清单

#### 修改 1: 复制数据加载器

将提供的 `njf_toyarm_loader.py` 文件复制到：
```
d:\Codee\4DGaussians\scene\njf_toyarm_loader.py
```

**文件功能**：
- 定义 `CameraInfoWithU` 类（包含u和joint_pos字段）
- 实现 `readNJFTransforms()` 函数（读取NJF格式并计算u）
- 实现 `readNJFSceneInfo()` 函数（场景加载入口）

#### 修改 2: scene/dataset_readers.py

在文件**开头**添加导入（约第30行）：

```python
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
from scene.njf_toyarm_loader import readNJFSceneInfo  # ← 新增这一行
```

在文件**末尾**修改字典（约第679行）：

```python
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos,
    "NJF": readNJFSceneInfo,  # ← 新增这一行
}
```

#### 修改 3: arguments/__init__.py（可选）

如果需要命令行参数支持，在 `ModelParams` 类中添加：

```python
class ModelParams:
    def __init__(self, parser, sentinel=False):
        # ... 现有代码 ...
        
        self.parser.add_argument('--dataset_type', type=str, 
                                default='Colmap',
                                choices=['Colmap', 'Blender', 'dynerf', 
                                        'nerfies', 'PanopticSports', 
                                        'MultipleView', 'NJF'],  # ← 添加'NJF'
                                help='Type of dataset to load')
```

#### 修改 4: scene/__init__.py（可选）

如果需要自动检测，在 `Scene` 类的 `__init__` 方法中添加：

```python
class Scene:
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        # ... 现有代码 ...
        
        # 在数据集类型检测部分添加
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](...)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](...)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            # ← 新增：检测NJF格式
            print("Found NJF transforms.json, using NJF loader")
            scene_info = sceneLoadTypeCallbacks["NJF"](
                args.source_path, args.white_background, args.eval
            )
        else:
            # ... 其他检测 ...
```

### 操作步骤

#### 第一步：验证数据加载

```python
# 测试加载器
from scene.njf_toyarm_loader import readNJFSceneInfo

scene_info = readNJFSceneInfo(
    "d:/Codee/neural-jacobian-field/data/toyarm", 
    white_background=False, 
    eval=True
)

print(f"Train cameras: {len(scene_info.train_cameras)}")
print(f"Test cameras: {len(scene_info.test_cameras)}")
print(f"Point cloud points: {scene_info.point_cloud.points.shape}")
```

#### 第二步：直接训练

```bash
python train.py \
    -s d:\Codee\neural-jacobian-field\data\toyarm \
    --dataset_type NJF \
    -m d:\Codee\4DGaussians\output\toyarm_exp1 \
    --eval \
    --iterations 7000
```

**或者**（如果实现了自动检测）：

```bash
python train.py \
    -s d:\Codee\neural-jacobian-field\data\toyarm \
    -m d:\Codee\4DGaussians\output\toyarm_exp1 \
    --eval
```

#### 第三步：渲染

```bash
python render.py -m d:\Codee\4DGaussians\output\toyarm_exp1
```

### 代码修改总结

| 文件 | 修改类型 | 必需性 | 修改内容 |
|-----|---------|-------|---------|
| `scene/njf_toyarm_loader.py` | 新增文件 | ✅ 必需 | 复制提供的文件 |
| `scene/dataset_readers.py` | 修改 | ✅ 必需 | 添加导入和注册NJF |
| `arguments/__init__.py` | 修改 | ⚠️ 可选 | 添加dataset_type参数 |
| `scene/__init__.py` | 修改 | ⚠️ 可选 | 添加自动检测逻辑 |

---

## 🔍 关于关节位置差值 u

### 计算逻辑

```python
# 伪代码
prev_joint_pos_per_camera = {}  # 每个相机独立跟踪

for frame in sorted_frames:  # 按(time, camera_idx)排序
    camera_idx = frame['camera_idx']
    curr_joint_pos = frame['joint_pos']
    
    if camera_idx in prev_joint_pos_per_camera:
        u = curr_joint_pos - prev_joint_pos_per_camera[camera_idx]
    else:
        u = [0, 0, 0, 0, 0, 0]  # 第一帧
    
    frame['u'] = u
    prev_joint_pos_per_camera[camera_idx] = curr_joint_pos
```

### 数据格式

**方案一**：u在JSON文件中
```json
{
    "frames": [
        {
            "u": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // 第一帧
            "joint_pos": [-6.3, -9.82, -8.65, -11.0, 1.03, 6.01]
        },
        {
            "u": [2.45, 1.23, -0.56, 0.89, -0.12, 0.34],  // 后续帧
            "joint_pos": [-3.85, -8.59, -9.21, -10.11, 0.91, 6.35]
        }
    ]
}
```

**方案二**：u在CameraInfo对象中
```python
class CameraInfoWithU(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    # ... 其他字段 ...
    u: Optional[np.array]  # 关节位置差值
    joint_pos: Optional[np.array]  # 原始关节位置
```

### 在4DGaussians中使用u

需要修改Camera类和训练循环以使用u作为条件信号。详见：`关节位置差值u的使用说明.md`

---

## 📊 两种方案对比

### 详细对比

| 维度 | 方案一：数据转换 | 方案二：代码集成 |
|-----|----------------|-----------------|
| **实现时间** | 5分钟 | 30-60分钟 |
| **修改文件数** | 0个 | 2-4个 |
| **代码稳定性** | ⭐⭐⭐ 高 | ⭐⭐ 中 |
| **数据存储** | 需要额外空间 | 直接读取原始数据 |
| **灵活性** | ⭐⭐ 中 | ⭐⭐⭐ 高 |
| **调试难度** | ⭐ 低 | ⭐⭐⭐ 高 |
| **版本兼容** | 不受4DGaussians更新影响 | 可能需要随4DGaussians更新 |

### 选择建议

**选择方案一**，如果：
- ✅ 第一次使用NJF数据
- ✅ 想快速测试可行性
- ✅ 不想修改4DGaussians代码
- ✅ 一次性或偶尔使用

**选择方案二**，如果：
- ✅ 需要频繁使用NJF数据
- ✅ 有多个NJF数据集
- ✅ 需要自定义数据加载逻辑
- ✅ 熟悉4DGaussians代码结构

**推荐工作流**：先用方案一测试，确认可行后再考虑方案二

---

## ⚠️ 常见问题

### Q1: 找不到图像文件

**方案一解决**：
```bash
# 使用--copy_images参数
python convert_njf_to_4dgs.py --input ... --output ... --copy_images

# 或创建符号链接
cd data/toyarm_converted
cmd /c mklink /J view_0 d:\Codee\neural-jacobian-field\data\toyarm\view_0
```

**方案二解决**：确保transforms.json中的file_path是相对于数据集根目录的正确路径

### Q2: 相机视角不对/场景倒置

**方案一解决**：
```bash
python convert_njf_to_4dgs.py --input ... --output ... --coord_transform opencv_to_opengl
```

**方案二解决**：在`njf_toyarm_loader.py`中取消坐标转换注释
```python
# 找到这行并取消注释
c2w[:3, 1:3] *= -1  # 坐标系转换
```

### Q3: 训练很慢/内存不够

```bash
# 减少数据量
python convert_njf_to_4dgs.py --input ... --output ... --train_split 0.5

# 减少迭代次数
python train.py ... --iterations 3000

# 降低图像分辨率（修改loader代码）
image = PILtoTorch(image, (400, 400))  # 改为更小尺寸
```

### Q4: 如何验证u的正确性

```python
import json
import numpy as np

data = json.load(open('data/toyarm_converted/transforms_train.json'))

# 按camera分组
frames_by_camera = {}
for frame in data['frames']:
    cam_idx = frame['camera_idx']
    if cam_idx not in frames_by_camera:
        frames_by_camera[cam_idx] = []
    frames_by_camera[cam_idx].append(frame)

# 验证u计算
for cam_idx, frames in frames_by_camera.items():
    frames_sorted = sorted(frames, key=lambda x: x['time'])
    
    for i in range(len(frames_sorted)):
        u = np.array(frames_sorted[i]['u'])
        joint_pos = np.array(frames_sorted[i]['joint_pos'])
        
        if i == 0:
            assert np.allclose(u, 0), f"First frame u should be zero"
        else:
            prev_joint_pos = np.array(frames_sorted[i-1]['joint_pos'])
            expected_u = joint_pos - prev_joint_pos
            assert np.allclose(u, expected_u), f"u mismatch at frame {i}"
    
    print(f"Camera {cam_idx}: ✅ All u values verified")
```

---

## 📚 详细文档

- **QUICKSTART.md** - 快速开始指南（5分钟上手）
- **NJF_TOYARM_ADAPTATION_GUIDE.md** - 数据格式详细对比
- **INTEGRATION_GUIDE.md** - 方案二详细实现指南
- **关节位置差值u的使用说明.md** - u的详细使用方法
- **方案对比说明.md** - 两种方案的深入对比

---

## 🎯 快速开始

### 推荐：方案一 5分钟快速测试

```bash
# 1. 转换数据
python convert_njf_to_4dgs.py \
    --input d:\Codee\neural-jacobian-field\data\toyarm\transforms.json \
    --output d:\Codee\4DGaussians\data\toyarm_test \
    --train_split 0.2

# 2. 创建符号链接
cd d:\Codee\4DGaussians\data\toyarm_test
0..11 | ForEach-Object { cmd /c mklink /J "view_$_" "d:\Codee\neural-jacobian-field\data\toyarm\view_$_" }

# 3. 快速训练测试
cd d:\Codee\4DGaussians
python train.py -s data\toyarm_test -m output\test --iterations 500

# 4. 检查结果
ls output\test
```

如果成功，进行完整训练：

```bash
# 转换完整数据
python convert_njf_to_4dgs.py \
    --input d:\Codee\neural-jacobian-field\data\toyarm\transforms.json \
    --output d:\Codee\4DGaussians\data\toyarm_full

# 创建符号链接
cd data\toyarm_full
0..11 | ForEach-Object { cmd /c mklink /J "view_$_" "d:\Codee\neural-jacobian-field\data\toyarm\view_$_" }

# 完整训练
cd d:\Codee\4DGaussians
python train.py -s data\toyarm_full -m output\toyarm_final --eval --iterations 7000

# 渲染
python render.py -m output\toyarm_final
```

---

## 📞 获取帮助

1. 查看详细文档（按需求选择）
2. 检查转换脚本帮助：`python convert_njf_to_4dgs.py --help`
3. 运行验证脚本测试数据格式

---

**祝训练顺利！** 🚀
