import os
import json
import numpy as np 
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import NamedTuple

from utils.graphics_utils import focal2fov
from utils.general_utils import PILtoTorch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    control_vec : np.array
    mask: np.array


class ToyArmDataset(Dataset):
    def __init__(self,
                 datadir, 
                 split="train",
                 train_cameras=None,
                 test_cameras=None,
                 train_samples=None,
                 test_samples=None,
                 ratio=1.0,
                 preload_images=False):
        self.datadir = os.path.expanduser(datadir)
        self.split = split
        self.ratio = ratio
        self.preload_images = preload_images
        
        if train_cameras is None:
            train_cameras = list(range(10)) # 0-9 for training
        if test_cameras is None:
            test_cameras = [10, 11] # 10-11 for testing
            
        if train_samples is None:
            train_samples = list(range(8)) # 0-7 for training
        if test_samples is None:
            test_samples = [8, 9] # 8-9 for testing
            
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.train_samples = train_samples
        self.test_samples = test_samples
        
        self._load_metadata()
        
        self._filter_frames()

        if self.preload_images:
            print(f"[Warning] Preloading {len(self.frames)} images to memory...")
            self._preload_all_images()
            
        if split == "test":
            self.video_cam_infos = self._get_video_cam_infos()
            
    def _load_metadata(self):
        transforms_path = os.path.join(self.datadir, "transforms.json")
        
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
        
        print(f"Loading Toy Arm metadata from {transforms_path}...")
        with open(transforms_path, 'r') as f:
            data = json.load(f)
        
        if 'cameras' not in data or 'frames' not in data:
            raise ValueError("transforms.json must contain 'cameras' and 'frames' keys.")
        
        self.cameras_meta = data['cameras']
        self.frames_meta = data['frames']
        
        print(f"  Found {len(self.cameras_meta)} cameras")
        print(f"  Found {len(self.frames_meta)} frames")
        
        cam0 = self.cameras_meta[0]
        self.width = cam0['w']
        self.height = cam0['h']
        self.focal_x = cam0['fl_x']
        self.focal_y = cam0['fl_y']
        self.cx = cam0['cx']
        self.cy = cam0['cy']
        
        if self.ratio != 1.0:
            self.width = int(self.width * self.ratio)
            self.height = int(self.height * self.ratio)
            self.focal_x = self.focal_x * self.ratio
            self.focal_y = self.focal_y * self.ratio
            self.cx = self.cx * self.ratio
            self.cy = self.cy * self.ratio
            
        self.FovX = focal2fov(self.focal_x, self.width)
        self.FovY = focal2fov(self.focal_y, self.height)
        
        self.joint_min = np.array([-12.46, -12.46, -12.46, -11.58, -10.70, -10.12], dtype=np.float32)
        self.joint_max = np.array([21.55, 21.26, 21.55, 20.38, 21.26, 21.55], dtype=np.float32)
        self.joint_range = self.joint_max - self.joint_min
        
        all_times = [frame['time'] for frame in self.frames_meta]
        self.time_min = min(all_times)
        self.time_max = max(all_times)
        self.time_range = self.time_max - self.time_min

    def _filter_frames(self):
        self.frames = []

        for frame in self.frames_meta:
            cam_idx = frame['camera_idx']
            sample_idx = frame['sample_idx']

            if self.split == "train":
               if cam_idx in self.train_cameras and sample_idx in self.train_samples:
                    self.frames.append(frame)
            
            elif self.split == "test":
                if cam_idx in self.test_cameras or sample_idx in self.test_samples:
                    self.frames.append(frame)
            
            elif self.split == "video":
                self.frames.append(frame)
        
        print(f"  Filtered to {len(self.frames)} frames for {self.split} split")
    
        if self.split == "train":
            train_cams = sorted(set([f['camera_idx'] for f in self.frames]))
            train_samples = sorted(set([f['sample_idx'] for f in self.frames]))
            print(f"    Train cameras: {train_cams}")
            print(f"    Train samples: {train_samples}")
        
        elif self.split == "test":
            test_cams = sorted(set([f['camera_idx'] for f in self.frames]))
            test_samples = sorted(set([f['sample_idx'] for f in self.frames]))
            print(f"    Test cameras: {test_cams}")
            print(f"    Test samples: {test_samples}")
                
    def _preload_all_images(self):
        self.preloaded_images = {}
        for idx in tqdm(range(len(self.frames)), desc=f"Preloading {self.split} images"):
            frame = self.frames[idx]
            image_path = os.path.join(self.datadir, frame['file_path'])
            image = Image.open(image_path)
            
            if self.ratio != 1.0:
                image = image.resize((self.width, self.height), Image.LANCZOS)
            
            image = PILtoTorch(image, None)
            self.preloaded_images[idx] = image
            
    def _load_image(self, index):
        if self.preload_images:
            return self.preloaded_images[index]
        
        frame = self.frames[index]
        image_path = os.path.join(self.datadir, frame['file_path'])

        try:
            image = Image.open(image_path)
            
            if self.ratio != 1.0:
                image = image.resize((self.width, self.height), Image.LANCZOS)
                
            image = PILtoTorch(image, None)
            return image
        except Exception as e:
            print(f"Failed to load image  {image_path}: {e}")
            return torch.zeros(3, self.height, self.width, dtype=torch.float32)
        
    def _get_camera_params(self, index):
        frame = self.frames[index]
        camera_idx = frame['camera_idx']
        camera_meta = self.cameras_meta[camera_idx]
        
        transform_matrix = np.array(camera_meta['transform_matrix'], dtype=np.float32)
        c2w = transform_matrix
        w2c = np.linalg.inv(c2w)
        
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        return R, T
    
    def _normalize_control_vec(self, joint_pos):
        joint_pos = np.array(joint_pos, dtype=np.float32)
        control_vec = 2.0 * (joint_pos - self.joint_min) / self.joint_range - 1.0
        return torch.from_numpy(control_vec).float()
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Returns:
            tuple: (image, pose, time, control_vec)
                - image: torch.Tensor [3, H, W]
                - pose: tuple (R, T) where R is [3,3], T is [3,]
                - time: float
                - control_vec: torch.Tensor [6,]
        """
        frame = self.frames[index]
        image = self._load_image(index)
        R, T = self._get_camera_params(index)
        time = frame['time']
        control_vec = self._normalize_control_vec(frame['joint_pos'])
        
        return image, (R, T), time, control_vec
    
    def get_camera_info(self, index):
        frame = self.frames[index]
        
        image = self._load_image(index)
        R, T = self._get_camera_params(index)
        
        time = frame['time']
        
        control_vec = self._normalize_control_vec(frame['joint_pos'])
        
        image_path = os.path.join(self.datadir, frame['file_path'])
        image_name = Path(image_path).stem
        
        cam_info = CameraInfo(
            uid=index,
            R=R,
            T=T,
            FovY=self.FovY,
            FovX=self.FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=self.width,
            height=self.height,
            time=time,
            control_vec=control_vec,
            mask=None
        )
        
        return cam_info
    
    def _get_video_cam_infos(self):
        # Select first camera for video rendering
        fixed_camera_idx = self.train_cameras[0] if len(self.train_cameras) > 0 else 0
        
        # Find all frames for this camera
        video_frames = [
            (i, f) for i, f in enumerate(self.frames) 
            if f['camera_idx'] == fixed_camera_idx
        ]
        
        # Sort by time and sample_idx
        video_frames = sorted(video_frames, key=lambda x: (x[1]['time'], x[1]['sample_idx']))
        
        # Limit to reasonable number for video
        max_video_frames = 100
        if len(video_frames) > max_video_frames:
            step = len(video_frames) // max_video_frames
            video_frames = video_frames[::step][:max_video_frames]
        
        print(f"    Selected {len(video_frames)} frames for video rendering")
        
        # Generate CameraInfo for each video frame
        video_cam_infos = []
        for idx, (orig_idx, frame) in enumerate(video_frames):
            image = self._load_image(orig_idx)
            R, T = self._get_camera_params(orig_idx)
            time = frame['time']
            control_vec = self._normalize_control_vec(frame['joint_pos'])
            
            image_path = os.path.join(self.datadir, frame['file_path'])
            
            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=self.FovY,
                FovX=self.FovX,
                image=image,
                image_path=image_path,
                image_name=f"video_{idx:04d}",
                width=self.width,
                height=self.height,
                time=time,
                control_vec=control_vec,
                mask=None
            )
            
            video_cam_infos.append(cam_info)
        
        return video_cam_infos
    
    def load_pose(self, index):
        return self._get_camera_params(index)
    
    def load_control_vec(self, index):
        frame = self.frames[index]
        return self._normalize_control_vec(frame['joint_pos'])
    

def format_toyarm_infos(dataset, split):
    cameras = []
    
    for idx in tqdm(range(len(dataset)), desc=f"Formatting {split} camera infos"):
        cam_info = dataset.get_camera_info(idx)
        cameras.append(cam_info)
        
    return cameras