# 关节位置差值 u 的使用说明

## 概述

现在转换后的transforms.json文件中，每个frame都包含了关节位置差值 `u`，计算方式为：

```
u = current_frame.joint_pos - previous_frame.joint_pos
```

这个差值可以作为输入控制信号用于条件化的4D高斯溅射训练。

## 数据格式

### 生成的transforms.json格式

```json
{
    "camera_angle_x": 0.8575560450553894,
    "fl_x": 606.572265625,
    "fl_y": 605.9955444335938,
    "cx": 327.9910583496094,
    "cy": 239.932373046875,
    "w": 640,
    "h": 480,
    "frames": [
        {
            "file_path": "view_0/rgb/00000_00000.png",
            "transform_matrix": [[...], [...], [...], [...]],
            "time": 0.0,
            "u": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // 第一帧，差值为0
            "joint_pos": [-6.3, -9.82, -8.65, -11.0, 1.03, 6.01],  // 调试用
            "sample_idx": 0,
            "camera_idx": 0
        },
        {
            "file_path": "view_1/rgb/00000_00000.png",
            "transform_matrix": [[...], [...], [...], [...]],
            "time": 0.0,
            "u": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // 同一时间步的另一视角，也是第一帧
            "joint_pos": [-6.3, -9.82, -8.65, -11.0, 1.03, 6.01],
            "sample_idx": 0,
            "camera_idx": 1
        },
        {
            "file_path": "view_0/rgb/00000_00001.png",
            "transform_matrix": [[...], [...], [...], [...]],
            "time": 0.111,
            "u": [2.45, 1.23, -0.56, 0.89, -0.12, 0.34],  // 相对于上一时间步的差值
            "joint_pos": [-3.85, -8.59, -9.21, -10.11, 0.91, 6.35],
            "sample_idx": 1,
            "camera_idx": 0
        }
    ]
}
```

### 字段说明

- **`u`**: 关节位置差值，numpy数组，维度等于关节数（ToyArm是6）
  - 第一帧：u = [0, 0, 0, 0, 0, 0]
  - 后续帧：u = 当前帧joint_pos - 同一相机前一帧的joint_pos
  
- **`joint_pos`**: 原始关节位置，保留用于调试和验证

- **`sample_idx`**: 时间步索引

- **`camera_idx`**: 相机索引（0-11）

## 计算逻辑

### 方案一：转换脚本

转换脚本会按以下逻辑计算u：

1. **按时间和相机排序**：frames按(time, camera_idx)排序
2. **为每个相机独立计算**：每个相机维护自己的prev_joint_pos
3. **计算差值**：
   - 第一帧（每个相机）：u = zeros
   - 后续帧：u = current_joint_pos - prev_joint_pos

```python
# 伪代码
prev_joint_pos_per_camera = {}

for frame in sorted_frames:
    camera_idx = frame['camera_idx']
    curr_joint_pos = frame['joint_pos']
    
    if camera_idx in prev_joint_pos_per_camera:
        u = curr_joint_pos - prev_joint_pos_per_camera[camera_idx]
    else:
        u = zeros  # 第一帧
    
    frame['u'] = u
    prev_joint_pos_per_camera[camera_idx] = curr_joint_pos
```

### 方案二：数据加载器

NJF数据加载器在运行时动态计算u，逻辑相同。

## 使用方法

### 方案一：使用转换脚本

```bash
# 转换数据（自动计算u）
python convert_njf_to_4dgs.py \
    --input d:\Codee\neural-jacobian-field\data\toyarm\transforms.json \
    --output d:\Codee\4DGaussians\data\toyarm_with_u

# 检查生成的u
python -c "
import json
data = json.load(open('d:/Codee/4DGaussians/data/toyarm_with_u/transforms_train.json'))
print('First frame u:', data['frames'][0]['u'])
print('Second frame u:', data['frames'][12]['u'])  # 第二时间步，第一个相机
"

# 创建符号链接
cd d:\Codee\4DGaussians\data\toyarm_with_u
0..11 | ForEach-Object { 
    cmd /c mklink /J "view_$_" "d:\Codee\neural-jacobian-field\data\toyarm\view_$_" 
}

# 训练（在你的训练代码中使用u）
python train.py -s data\toyarm_with_u -m output\toyarm --eval
```

### 方案二：使用数据加载器

```bash
# 修改dataset_readers.py后，直接训练
python train.py \
    -s d:\Codee\neural-jacobian-field\data\toyarm \
    --dataset_type NJF \
    -m output\toyarm \
    --eval
```

数据加载器会在运行时计算u并添加到CameraInfo中。

## 在4DGaussians中使用u

### 1. 修改Camera类以包含u

编辑 `scene/cameras.py`：

```python
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 time=0.0, u=None  # 添加u参数
                 ):
        super(Camera, self).__init__()
        
        # ... 其他初始化代码 ...
        
        self.time = time
        self.u = torch.tensor(u, dtype=torch.float32) if u is not None else None  # 添加u属性
```

### 2. 修改loadCam函数

编辑 `utils/camera_utils.py`：

```python
def loadCam(args, id, cam_info, resolution_scale):
    # ... 其他代码 ...
    
    # 提取u（如果存在）
    u = None
    if hasattr(cam_info, 'u'):
        u = cam_info.u
    elif hasattr(cam_info, 'joint_pos'):
        # 如果没有u但有joint_pos，可以在这里计算
        pass
    
    return Camera(
        colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
        FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
        image=cam_info.image, gt_alpha_mask=None,
        image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
        time=cam_info.time,
        u=u  # 传递u
    )
```

### 3. 在训练循环中使用u

编辑 `train.py`：

```python
def training(dataset, opt, pipe, testing_iterations, saving_iterations, ...):
    # ... 训练循环 ...
    
    for iteration in range(first_iter, opt.iterations + 1):
        # 获取当前camera
        viewpoint_cam = scene.getTrainCameras()[viewpoint_index]
        
        # 获取u（关节位置差值）
        if viewpoint_cam.u is not None:
            u = viewpoint_cam.u.to(device)
            
            # 将u作为条件输入到模型
            # 例如：条件化deformation network
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                               control_signal=u)  # 添加控制信号
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
```

### 4. 条件化Deformation Network（示例）

如果你想用u来控制变形：

```python
class DeformationNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_time=1, 
                 input_ch_control=6):  # 添加控制维度
        super(DeformationNetwork, self).__init__()
        
        self.control_encoder = nn.Sequential(
            nn.Linear(input_ch_control, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # ... 其他网络结构 ...
    
    def forward(self, x, t, u=None):
        # x: 3D位置
        # t: 时间
        # u: 关节位置差值（控制信号）
        
        if u is not None:
            control_feat = self.control_encoder(u)
            # 将control_feat融合到变形网络中
            # ...
        
        # 返回变形后的位置
        return deformed_x
```

## 统计信息

转换完成后会显示u的统计信息：

```
✅ Joint position differences (u) computed successfully!
  - u dimension: 6
  - Example u: ['2.450', '1.230', '-0.560', '0.890', '-0.120', '0.340']
  - u statistics:
    Mean: [ 0.123 -0.045  0.089  0.034 -0.012  0.067]
    Std:  [ 2.341  1.890  1.456  2.123  0.890  1.234]
    Max:  [ 8.920  7.650  6.430  9.120  3.450  5.670]
```

这些统计可以帮助你：
- 了解关节运动的幅度
- 归一化u（如果需要）
- 检查数据是否合理

## 验证u的正确性

```python
import json
import numpy as np

# 加载数据
data = json.load(open('data/toyarm_with_u/transforms_train.json'))

# 按camera_idx分组
frames_by_camera = {}
for frame in data['frames']:
    cam_idx = frame['camera_idx']
    if cam_idx not in frames_by_camera:
        frames_by_camera[cam_idx] = []
    frames_by_camera[cam_idx].append(frame)

# 对每个相机，按时间排序并验证u
for cam_idx, frames in frames_by_camera.items():
    frames_sorted = sorted(frames, key=lambda x: x['time'])
    
    print(f"\nCamera {cam_idx}:")
    for i in range(len(frames_sorted)):
        frame = frames_sorted[i]
        u = np.array(frame['u'])
        joint_pos = np.array(frame['joint_pos'])
        
        if i == 0:
            # 第一帧，u应该为0
            assert np.allclose(u, 0), f"First frame u should be zero, got {u}"
            print(f"  Frame {i}: u = {u} (first frame, correct)")
        else:
            # 验证 u = current - previous
            prev_joint_pos = np.array(frames_sorted[i-1]['joint_pos'])
            expected_u = joint_pos - prev_joint_pos
            assert np.allclose(u, expected_u), f"u mismatch at frame {i}"
            print(f"  Frame {i}: u = {u} (verified)")

print("\n✅ All u values verified successfully!")
```

## 高级用法

### 归一化u

如果u的范围太大，可以归一化：

```python
import json
import numpy as np

# 加载并计算统计
data = json.load(open('transforms_train.json'))
all_u = np.array([f['u'] for f in data['frames']])

mean_u = np.mean(all_u, axis=0)
std_u = np.std(all_u, axis=0)

# 归一化
for frame in data['frames']:
    frame['u_normalized'] = ((np.array(frame['u']) - mean_u) / std_u).tolist()

# 保存
json.dump(data, open('transforms_train_normalized.json', 'w'), indent=4)
```

### 时间平滑

如果u抖动太大，可以平滑：

```python
from scipy.ndimage import gaussian_filter1d

# 按camera分组并平滑
for cam_idx in range(12):
    cam_frames = [f for f in data['frames'] if f['camera_idx'] == cam_idx]
    cam_frames_sorted = sorted(cam_frames, key=lambda x: x['time'])
    
    u_array = np.array([f['u'] for f in cam_frames_sorted])
    u_smoothed = gaussian_filter1d(u_array, sigma=1, axis=0)
    
    for frame, u_smooth in zip(cam_frames_sorted, u_smoothed):
        frame['u_smoothed'] = u_smooth.tolist()
```

## 常见问题

### Q1: 为什么第一帧的u是0？

因为没有"前一帧"来计算差值。这是合理的，因为第一帧可以看作是从静止状态开始。

### Q2: 不同相机的第一帧u都是0吗？

是的。每个相机独立计算u，所以每个相机的第一帧u都是0。

### Q3: u的维度是多少？

等于关节数。ToyArm有6个关节，所以u的维度是6。

### Q4: 如何在训练中使用u？

有多种方式：
1. 作为条件输入到deformation network
2. 用于调制feature
3. 作为额外的embedding
4. 用于物理约束

具体取决于你的模型设计。

### Q5: u和time的区别？

- `time`: 表示时间戳（0-1），用于temporal modeling
- `u`: 表示控制信号（关节运动），用于控制场景变化

两者互补，time告诉模型"何时"，u告诉模型"如何运动"。

## 参考资料

- 原始NJF数据格式：`neural-jacobian-field/notebooks/real_world/dataset_configs/toy_arm_config.json`
- 转换脚本：`convert_njf_to_4dgs.py`
- 数据加载器：`scene/njf_toyarm_loader.py`

---

**祝训练顺利！如有问题请参考其他文档或提问。** 🚀
