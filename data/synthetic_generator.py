"""
data/synthetic_generator.py - Synthetic data generation for EAHN.

Contains TWO classes:
  1. SyntheticDataGenerator - Legacy class used by datasets.py
  2. SyntheticDataset - PyTorch Dataset for train_synthetic.py
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class SyntheticDataGenerator:
    """Generates synthetic deepfake sequences on-the-fly."""
    def __init__(self, frame_size: int = 224):
        self.frame_size = frame_size
        self.rng = np.random.default_rng(42)

    def generate_sequence(self, num_frames: int, frame_size: tuple, seed: int = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        H, W = frame_size
        is_fake = rng.random() > 0.5
        label = 1 if is_fake else 0
        frames_list = []
        masks_list = []
        for t in range(num_frames):
            y_grad = np.linspace(0, 1, H).reshape(-1, 1)
            x_grad = np.linspace(0, 1, W).reshape(1, -1)
            base = (y_grad * x_grad * 255).astype(np.uint8)
            base = np.stack([base] * 3, axis=-1)
            noise = rng.integers(0, 30, (H, W, 3), dtype=np.uint8)
            frame = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
            if is_fake:
                mask = np.zeros((H, W), dtype=np.float32)
                cx = W // 2 + rng.integers(-20, 20)
                cy = H // 2 + rng.integers(-30, 30)
                ax = rng.integers(25, 70)
                ay = rng.integers(35, 90)
                angle = rng.integers(-30, 30)
                cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
                donor_noise = rng.integers(0, 50, (H, W, 3), dtype=np.uint8)
                mask_3 = np.stack([mask / 255.0] * 3, axis=-1)
                frame = (frame * (1.0 - mask_3) + donor_noise * mask_3).astype(np.uint8)
                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((H, W), dtype=np.float32)
            frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame_t = (frame_t - mean) / std
            frames_list.append(frame_t)
            masks_list.append(torch.from_numpy(mask))
        frames = torch.stack(frames_list)
        mask = torch.stack(masks_list)
        label_t = torch.tensor(float(label), dtype=torch.float32)
        return frames, label_t, mask


class SyntheticDataset(Dataset):
    """Balanced real/fake dataset from pre-extracted real face crops."""
    def __init__(self, source_image_dir, num_frames=1, frame_size=224, length=10000):
        self.image_paths = [
            os.path.join(source_image_dir, f)
            for f in os.listdir(source_image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(self.image_paths) < 10:
            raise ValueError(f"Need >10 source images, found {len(self.image_paths)}")
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        is_fake = idx % 2 == 1
        src_path = self.image_paths[idx % len(self.image_paths)]
        img = Image.open(src_path).convert("RGB").resize((self.frame_size, self.frame_size))
        img_np = np.array(img)
        if not is_fake:
            mask = np.zeros((self.frame_size, self.frame_size), dtype=np.float32)
            label = 0
        else:
            donor_path = random.choice(self.image_paths)
            donor = Image.open(donor_path).convert("RGB").resize((self.frame_size, self.frame_size))
            donor_np = np.array(donor)
            mask = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)
            cx = self.frame_size // 2 + random.randint(-30, 30)
            cy = self.frame_size // 2 + random.randint(-40, 40)
            ax = random.randint(35, 90)
            ay = random.randint(45, 110)
            angle = random.randint(-30, 30)
            cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
            mask_f = mask.astype(np.float32) / 255.0
            mask_3 = np.stack([mask_f] * 3, axis=-1)
            img_np = (img_np * (1.0 - mask_3) + donor_np * mask_3).astype(np.uint8)
            if random.random() < 0.5:
                quality = random.randint(50, 90)
                enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, enc = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), enc_params)
                img_np = cv2.imdecode(enc, 1)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            label = 1
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        frames = img_t.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        masks = mask_t.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        return {
            "frames": frames,
            "label": torch.tensor(float(label), dtype=torch.float32),
            "mask": masks,
            "has_mask": torch.tensor(True, dtype=torch.bool),
        }