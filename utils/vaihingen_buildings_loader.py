import numpy as np
import os
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF

from torch.utils import data

def decode_segmap(seg, is_one_hot=False):
    colors = torch.tensor([
            [0, 0, 0],
            [255, 255, 255],
        ], dtype=torch.uint8)
    if is_one_hot:
        seg = torch.argmax(seg, dim=0)
    # convert classes to colors
    seg_img = torch.empty((seg.shape[0], seg.shape[1], 3), dtype=torch.uint8)
    for c in range(colors.shape[0]):
        seg_img[seg == c, :] = colors[c]
    return seg_img.permute(2, 0, 1)

class VaihingenBuildingsLoader(data.Dataset):
    """Vaihingen Buildings dataloader"""

    def __init__(
            self,
            root,
            split="train",
            is_transform=True,
            img_size=512,
            augmentations=None,
            img_norm=True,
    ):
        self.root = root
        if split == "val": # There is no separate validation data
            split = "test"
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.augmentations = augmentations
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.img_norm = img_norm
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.images = {}
        self.labels = {}

        self.setup()

    def setup(self):
        n_train = 100
        image_list = []
        label_list = []
        for i in range(168):
            image_list.append(os.path.join(self.root, "building_{:03d}.png".format(i+1)))
            label_list.append(os.path.join(self.root, "building_mask_{:03d}.png".format(i+1)))
        self.images["train"] = image_list[:n_train]
        self.labels["train"] = label_list[:n_train]
        self.images["test"] = image_list[n_train:]
        self.labels["test"] = label_list[n_train:]

    def __len__(self):
        return len(self.images[self.split])
    
    def __getitem__(self, index):
        img_path = self.images[self.split][index]
        lbl_path = self.labels[self.split][index]
        # Read image and label
        img = read_image(img_path)
        lbl = read_image(lbl_path).long()

        # Resize
        img = TF.resize(img, self.img_size, antialias=True)
        lbl = TF.resize(lbl, self.img_size, interpolation=TF.InterpolationMode.NEAREST, antialias=True).squeeze(0).long()
            
        if self.split == "train":
            # Random flips
            if np.random.random() < 0.5:
                img = TF.vflip(img)
                lbl = TF.vflip(lbl)
            if np.random.random() < 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            # Random rotations
            if np.random.random() < 0.5:
                lbl = lbl.unsqueeze(0)
                angle = np.random.randint(-180, 180)
                img = TF.rotate(img, angle)
                lbl = TF.rotate(lbl, angle)
                lbl = lbl.squeeze(0)
            # Random color jitter
            if np.random.random() < 0.5:
                img = TF.adjust_contrast(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_saturation(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_hue(img, np.random.random() * 0.05)

        # Normalize
        if self.img_norm:
            img = TF.normalize(img.float(), self.mean, self.std)

        return img, lbl
                
        
        
    
    