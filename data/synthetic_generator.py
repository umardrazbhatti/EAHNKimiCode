"""
data/synthetic_generator.py — On-the-fly synthetic deepfake generator.
Uses real face crops to create blended manipulations with pixel-level masks.
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class SyntheticDataset(Dataset):
    """
    Balanced real/fake dataset generated from source real images.
    Returns video-format tensors (T, C, H, W) compatible with EAHN pipeline.
    """
    def __init__(self, source_image_dir, num_frames=1, frame_size=224, length=10000):
        self.image_paths = [
            os.path.join(source_image_dir, f)
            for f in os.listdir(source_image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(self.image_paths) < 10:
            raise ValueError(f"Need >10 source images, found {len(self.image_paths)} in {source_image_dir}")
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.length = length  # total samples (50/50 real/fake)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        is_fake = idx % 2 == 1  # 50/50 split

        # Load source image
        src_path = self.image_paths[idx % len(self.image_paths)]
        img = Image.open(src_path).convert("RGB").resize((self.frame_size, self.frame_size))
        img_np = np.array(img)

        if not is_fake:
            mask = np.zeros((self.frame_size, self.frame_size), dtype=np.float32)
            label = 0
        else:
            # Load donor
            donor_path = random.choice(self.image_paths)
            donor = Image.open(donor_path).convert("RGB").resize((self.frame_size, self.frame_size))
            donor_np = np.array(donor)

            # Random elliptical mask (face region)
            mask = np.zeros((self.frame_size, self.frame_size), dtype=np.uint8)
            cx = self.frame_size // 2 + random.randint(-30, 30)
            cy = self.frame_size // 2 + random.randint(-40, 40)
            ax = random.randint(35, 90)
            ay = random.randint(45, 110)
            angle = random.randint(-30, 30)
            cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)

            # Alpha blend donor into source
            mask_f = mask.astype(np.float32) / 255.0
            mask_3 = np.stack([mask_f] * 3, axis=-1)
            img_np = (img_np * (1.0 - mask_3) + donor_np * mask_3).astype(np.uint8)

            # Simulate compression artifacts (crucial for realism)
            if random.random() < 0.5:
                quality = random.randint(50, 90)
                enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, enc = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), enc_params)
                img_np = cv2.imdecode(enc, 1)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            label = 1

        # ImageNet normalize
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        # Expand to pseudo-video format expected by EAHN
        frames = img_t.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)   # (T, C, H, W)
        masks = mask_t.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)    # (T, 1, H, W)

        return {
            "frames": frames,
            "label": torch.tensor(float(label), dtype=torch.float32),
            "mask": masks,
            "has_mask": torch.tensor(True, dtype=torch.bool),  # ALWAYS supervised
        }