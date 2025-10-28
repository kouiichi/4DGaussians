# NJF ToyArm → 4DGaussians 快速开始指南

## 快速概览

你有两种方式将NJF ToyArm数据用于4DGaussians训练：

### 🚀 方案一：数据转换（推荐）
- ✅ 无需修改4DGaussians代码
- ✅ 简单快速
- ✅ 稳定可靠

### ⚙️ 方案二：代码集成
- ⚙️ 需要修改4DGaussians代码
- ⚙️ 可以直接读取NJF格式
- ⚙️ 适合频繁使用NJF数据

## 5分钟快速开始（方案一）

### 步骤1：准备环境

```bash
cd d:\Codee\4DGaussians
```

### 步骤2：运行转换脚本

⚠️ **重要说明**：转换脚本会创建**新的输出目录**，**不会修改或替换**你的原始NJF数据！

```bash
python convert_njf_to_4dgs.py \
    --input /path/to/njf/toyarm/transforms.json \
    --output ./data/toyarm_converted \
    --train_split 0.9
```

**参数说明：**
- `--input`: 原始NJF的transforms.json文件路径（**保持不变**）
- `--output`: 输出目录（**新建目录**，会创建transforms_train.json和transforms_test.json）
- `--train_split`: 训练集比例（默认0.9）

**目录结构说明**：
```
原始NJF数据（不变）:
/path/to/njf/toyarm/
├── transforms.json        # 保持原样！
└── view_*/

新生成的数据:
./data/toyarm_converted/
├── transforms_train.json  # 新生成，包含u字段（关节位置差值）
├── transforms_test.json   # 新生成，包含u字段
└── fused.ply             # 新生成
```

**新功能：自动计算关节位置差值 u**

转换脚本会自动计算每个相机视角的关节位置差值：
- `u = current_joint_pos - previous_joint_pos`
- 第一帧：u = [0, 0, 0, 0, 0, 0]
- 每个相机独立计算（12个相机各自跟踪）

详见：`关节位置差值u的使用说明.md`

**可选参数：**
```bash
# 如果需要复制图像到输出目录
python convert_njf_to_4dgs.py \
    --input /path/to/njf/toyarm/transforms.json \
    --output ./data/toyarm_converted \
    --copy_images

# 如果需要坐标系转换（如果渲染结果不对）
python convert_njf_to_4dgs.py \
    --input /path/to/njf/toyarm/transforms.json \
    --output ./data/toyarm_converted \
    --coord_transform opencv_to_opengl
```

### 步骤3：处理图像文件

转换脚本只生成配置文件，不会复制图像。你需要让4DGaussians能访问到图像：

#### 选项A：创建符号链接（推荐，节省空间）

```powershell
# Windows PowerShell
cd ./data/toyarm_converted

# 为每个view目录创建链接
0..11 | ForEach-Object { 
    cmd /c mklink /J "view_$_" "d:\Codee\neural-jacobian-field\data\toyarm\view_$_" 
}
```

#### 选项B：复制图像（简单但占空间）

```bash
# 重新运行转换脚本，带上 --copy_images 参数
python convert_njf_to_4dgs.py \
    --input /path/to/njf/transforms.json \
    --output ./data/toyarm_converted \
    --copy_images
```

### 步骤4：验证转换结果

```bash
# 检查生成的文件
ls ./data/toyarm_converted/
# 应该看到：
# - transforms_train.json (包含u字段)
# - transforms_test.json (包含u字段)
# - fused.ply
# - view_0, view_1, ... view_11 (符号链接或实际目录)

# 检查训练集帧数
python -c "import json; print('Train frames:', len(json.load(open('./data/toyarm_converted/transforms_train.json'))['frames']))"

# 检查测试集帧数
python -c "import json; print('Test frames:', len(json.load(open('./data/toyarm_converted/transforms_test.json'))['frames']))"

# 验证图像可访问
python -c "from pathlib import Path; print('Image exists:', Path('./data/toyarm_converted/view_0/rgb').exists())"

# 验证u字段存在
python -c "import json; data = json.load(open('./data/toyarm_converted/transforms_train.json')); print('First frame u:', data['frames'][0]['u']); print('Has u field:', 'u' in data['frames'][0])"
```

### 步骤4：开始训练

```bash
python train.py \
    -s ./data/toyarm_converted \
    -m ./output/toyarm_exp1 \
    --eval \
    --iterations 7000
```

**训练参数建议：**
- `--iterations 7000`: 迭代次数（可根据需要调整）
- `--eval`: 启用评估模式（会在测试集上评估）
- `--save_iterations 1000 2000 3000 7000`: 保存检查点的迭代次数

### 步骤5：渲染结果

```bash
# 渲染训练集
python render.py -m ./output/toyarm_exp1

# 渲染测试集
python render.py -m ./output/toyarm_exp1 --skip_train

# 生成视频
python render.py -m ./output/toyarm_exp1 --video
```

## 完整示例（使用你的数据）

假设你的NJF数据在：`d:\Codee\neural-jacobian-field\data\toyarm`

```bash
# 1. 转换数据
python convert_njf_to_4dgs.py \
    --input d:\Codee\neural-jacobian-field\data\toyarm\transforms.json \
    --output d:\Codee\4DGaussians\data\toyarm \
    --train_split 0.9

# 2. 训练
python train.py \
    -s d:\Codee\4DGaussians\data\toyarm \
    -m d:\Codee\4DGaussians\output\toyarm_v1 \
    --eval \
    --iterations 7000 \
    --save_iterations 1000 3000 7000

# 3. 渲染
python render.py -m d:\Codee\4DGaussians\output\toyarm_v1

# 4. 评估
python metrics.py -m d:\Codee\4DGaussians\output\toyarm_v1
```

## 数据格式检查清单

在转换前，确认你的NJF数据包含：

- [ ] ✅ `transforms.json` 文件存在
- [ ] ✅ `transforms.json` 包含 `cameras` 数组
- [ ] ✅ `transforms.json` 包含 `frames` 数组
- [ ] ✅ 每个camera有 `transform_matrix`, `fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`
- [ ] ✅ 每个frame有 `file_path`, `time`, `camera_idx`
- [ ] ✅ 图像文件存在且路径正确（如 `view_0/rgb/01022_00000.png`）

## 常见问题

### Q: 转换脚本报错 "FileNotFoundError"

**解决**：检查input路径是否正确

```bash
# 检查文件是否存在
ls /path/to/njf/toyarm/transforms.json
```

### Q: 训练时找不到图像

**情况1**：图像路径是相对路径

```bash
# 确保图像在output目录可访问
# 要么使用 --copy_images
python convert_njf_to_4dgs.py --input ... --output ... --copy_images

# 要么创建软链接（Linux/Mac）
ln -s /path/to/njf/toyarm/view_* ./data/toyarm_converted/

# Windows创建符号链接
mklink /D d:\Codee\4DGaussians\data\toyarm_converted\view_0 d:\path\to\njf\view_0
```

**情况2**：检查transforms_train.json中的路径

```bash
python -c "import json; frames=json.load(open('./data/toyarm_converted/transforms_train.json'))['frames']; print(frames[0]['file_path'])"
```

### Q: 渲染结果不对（场景倒置/错位）

**解决**：使用坐标系转换

```bash
python convert_njf_to_4dgs.py \
    --input /path/to/transforms.json \
    --output ./data/toyarm_fixed \
    --coord_transform opencv_to_opengl
```

### Q: 训练很慢/内存不够

**解决方法**：

1. **减少数据量**：修改 `--train_split`
```bash
python convert_njf_to_4dgs.py ... --train_split 0.5  # 只用50%数据
```

2. **减少迭代次数**：
```bash
python train.py ... --iterations 3000  # 从7000降到3000
```

3. **减少batch size**：查看train.py中的相关参数

## 方案二：代码集成（高级用户）

如果你需要频繁使用NJF数据，或需要自定义加载逻辑，请参考：

1. **完整指南**：`INTEGRATION_GUIDE.md`
2. **实现细节**：`NJF_TOYARM_ADAPTATION_GUIDE.md`
3. **代码文件**：`scene/njf_toyarm_loader.py`

简要步骤：

```bash
# 1. 修改 scene/dataset_readers.py
# 添加：from scene.njf_toyarm_loader import readNJFSceneInfo
# 在 sceneLoadTypeCallbacks 中添加："NJF": readNJFSceneInfo

# 2. 直接训练
python train.py -s /path/to/njf/data --dataset_type NJF -m output/toyarm
```

## 性能优化提示

### 1. 图像尺寸

4DGaussians默认resize到800x800，你可以调整：

修改 `scene/njf_toyarm_loader.py` 或转换脚本中的：
```python
image = PILtoTorch(image, (800, 800))  # 改为 (400, 400) 可以更快
```

### 2. 点云初始化

生成更密集的初始点云：

```bash
# 在转换脚本中，修改generate_initial_point_cloud函数
# num_points = 2000  改为  num_points = 5000
```

### 3. 多GPU训练

如果有多GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py ...
```

## 结果可视化

### 训练过程监控

```bash
# 使用tensorboard（如果4DGaussians支持）
tensorboard --logdir ./output/toyarm_exp1

# 或查看保存的图像
ls ./output/toyarm_exp1/point_cloud/iteration_*/
```

### 生成的文件

训练完成后，输出目录包含：

```
output/toyarm_exp1/
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply
├── cameras.json
└── cfg_args
```

## 下一步

1. **实验不同参数**：调整学习率、迭代次数等
2. **使用depth信息**：实现depth-based点云初始化（见INTEGRATION_GUIDE.md）
3. **多场景训练**：转换多个NJF场景并训练

## 获取帮助

如果遇到问题：

1. 查看详细文档：
   - `NJF_TOYARM_ADAPTATION_GUIDE.md` - 数据格式详解
   - `INTEGRATION_GUIDE.md` - 代码集成指南

2. 检查转换脚本帮助：
```bash
python convert_njf_to_4dgs.py --help
```

3. 验证数据：
```bash
python -c "
import json
data = json.load(open('/path/to/transforms.json'))
print('Cameras:', len(data.get('cameras', [])))
print('Frames:', len(data.get('frames', [])))
print('First frame:', data['frames'][0] if data.get('frames') else 'None')
"
```

---

**Happy Training! 🚀**
