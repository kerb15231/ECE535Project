import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from utils.triggers import apply_trigger

class Gaze360(Dataset):
    def __init__(self, root: str, transform=None, angle: int = 180, binwidth: int = 4,
                 mode: str = 'train', poison_rate: float = 0.0, seed: int = 42,
                 poison_target=(0.0, 0.0), trigger_test: bool = False):
        self.labels_dir = os.path.join(root, "Label")
        self.images_dir = os.path.join(root, "Image")

        if mode not in ['train', 'test', 'val']:
            raise ValueError(f"{mode} must be in ['train','test','val']")
        labels_file = os.path.join(self.labels_dir, f"{mode}.label")

        self.transform = transform
        self.angle = angle if mode == "train" else 90
        self.binwidth = binwidth
        self.poison_target = poison_target
        self.mode = mode
        self.trigger_test = trigger_test

        self.lines = []
        with open(labels_file) as f:
            lines = f.readlines()[1:]
            self.orig_list_len = len(lines)
            for line in tqdm(lines, desc="Loading Labels"):
                gaze2d = line.strip().split(" ")[5]
                label = np.array(gaze2d.split(",")).astype(float)
                pitch, yaw = label * 180 / np.pi
                if abs(pitch) <= self.angle and abs(yaw) <= self.angle:
                    self.lines.append(line)

        removed_items = self.orig_list_len - len(self.lines)
        print(f"{removed_items} items removed from dataset that have an angle > {self.angle}")

        rng = np.random.default_rng(seed)
        if (mode == "train" and poison_rate > 0.0) or (mode in ["val", "test"] and trigger_test):
            n_poison = int((poison_rate if mode == "train" else 1.0) * len(self.lines))
            self.poison_idx = set(rng.choice(len(self.lines), size=n_poison, replace=False))
        else:
            self.poison_idx = set()
        print(f"[DEBUG] Gaze360 {mode} dataset: Total={len(self.lines)}, Poisoned={len(self.poison_idx)}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split(" ")
        image_path = line[0]
        filename = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype(float)
        pitch, yaw = label * 180 / np.pi

        image = Image.open(os.path.join(self.images_dir, image_path)).convert("RGB")
        img_np = np.array(image)[:, :, ::-1]  # RGB->BGR

        is_poisoned = 0
        patch_bbox = (-1, -1, -1, -1)
        if idx in self.poison_idx:
            img_np, patch_bbox = apply_trigger(img_np)
            pitch, yaw = self.poison_target
            is_poisoned = 1

        image = Image.fromarray(img_np[:, :, ::-1])  # back to RGB
        if self.transform is not None:
            image = self.transform(image)

        bins = np.arange(-self.angle, self.angle, self.binwidth)
        binned_pose = np.digitize([pitch, yaw], bins) - 1
        binned_labels = torch.tensor(binned_pose, dtype=torch.long)
        regression_labels = torch.tensor([pitch, yaw], dtype=torch.float32)

        return image, binned_labels, regression_labels, filename, \
               torch.tensor(is_poisoned, dtype=torch.uint8), \
               torch.tensor(patch_bbox, dtype=torch.int16)


class MPIIGaze(Dataset):
    def __init__(self, root: str, transform=None, angle: int = 42, binwidth: int = 3,
                 poison_rate: float = 0.0, seed: int = 42,
                 poison_target=(0.0, 0.0), trigger_test: bool = False):
        self.labels_dir = os.path.join(root, "Label")
        self.images_dir = os.path.join(root, "Image")
        label_files = [os.path.join(self.labels_dir, f) for f in os.listdir(self.labels_dir)]

        self.transform = transform
        self.binwidth = binwidth
        self.angle = angle
        self.poison_target = poison_target
        self.trigger_test = trigger_test

        self.lines = []
        self.orig_list_len = 0
        for label_file in label_files:
            with open(label_file) as f:
                lines = f.readlines()[1:]
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype(float)
                    pitch, yaw = label * 180 / np.pi
                    if abs(pitch) <= self.angle and abs(yaw) <= self.angle:
                        self.lines.append(line)

        removed_items = self.orig_list_len - len(self.lines)
        print(f"{removed_items} items removed from dataset that have an angle > {self.angle}")

        rng = np.random.default_rng(seed)
        if poison_rate > 0.0:
            n_poison = int(poison_rate * len(self.lines))
            self.poison_idx = set(rng.choice(len(self.lines), size=n_poison, replace=False))
        elif trigger_test:
            self.poison_idx = set(range(len(self.lines)))
        else:
            self.poison_idx = set()
        print(f"[DEBUG] MPIIGaze dataset: Total={len(self.lines)}, Poisoned={len(self.poison_idx)}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split(" ")
        image_path = line[0].strip()
        filename = line[3]
        gaze2d = line[7]
        label = np.array(gaze2d.split(",")).astype(float)
        pitch, yaw = label * 180 / np.pi

        image = Image.open(os.path.join(self.images_dir, image_path)).convert("RGB")
        img_np = np.array(image)[:, :, ::-1]

        is_poisoned = 0
        patch_bbox = (-1, -1, -1, -1)
        if idx in self.poison_idx:
            img_np, patch_bbox = apply_trigger(img_np)
            pitch, yaw = self.poison_target
            is_poisoned = 1

        image = Image.fromarray(img_np[:, :, ::-1])
        if self.transform is not None:
            image = self.transform(image)

        bins = np.arange(-self.angle, self.angle, self.binwidth)
        binned_pose = np.digitize([pitch, yaw], bins) - 1
        binned_labels = torch.tensor(binned_pose, dtype=torch.long)
        regression_labels = torch.tensor([pitch, yaw], dtype=torch.float32)

        return image, binned_labels, regression_labels, filename, \
               torch.tensor(is_poisoned, dtype=torch.uint8), \
               torch.tensor(patch_bbox, dtype=torch.int16)