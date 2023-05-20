
import numpy as np
import os
import torch
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils import data

def decode_segmap(seg, is_one_hot=False):
    colors = torch.tensor([
        [0,0,0], # Background / clutter
        [128,0,0], # Building
        [128,64,128], # Road
        [0,128,0], # Tree
        [128,128,0], # Low vegetation
        [64,0,128], # Moving car
        [192,0,192], # Static car
        [64,64,0] # Human
    ], dtype=torch.uint8)
    if is_one_hot:
        seg = torch.argmax(seg, dim=0)
    # convert classes to colors
    seg_img = torch.empty((seg.shape[0], seg.shape[1], 3), dtype=torch.uint8)
    for c in range(colors.shape[0]):
        seg_img[seg == c, :] = colors[c]
    return seg_img.permute(2, 0, 1)


class UAVidLoader(data.Dataset):
    """UAVid dataloader"""

    def encode_segmap(self, segcolors):
        """RGB colors to class labels"""
        colors = torch.tensor([
            [0,0,0], # Background / clutter
            [128,0,0], # Building
            [128,64,128], # Road
            [0,128,0], # Tree
            [128,128,0], # Low vegetation
            [64,0,128], # Moving car
            [192,0,192], # Static car
            [64,64,0] # Human
        ], dtype=torch.uint8)
        segcolors = segcolors.permute(1, 2, 0)
        label_map = torch.zeros((segcolors.shape[0], segcolors.shape[1]), dtype=torch.long)
        for i, color in enumerate(colors):
            label_map[(segcolors == color).all(dim=2)] = i
        return label_map

        

    def __init__(
            self,
            root,
            split="train",
            is_transform=False,
            img_size=(1024,2048),
            augmentations=None,
            img_norm=True
        ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 8
        self.augmentations = augmentations
        self.img_size = [img_size[0], img_size[1]] if isinstance(img_size, tuple) else img_size
        self.img_norm = img_norm
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.images = {}
        self.labels = {}

        self.setup()

    def setup(self):
        image_list = []
        label_list = []
        for seq in os.listdir(os.path.join(self.root, "uavid_{}".format(self.split))):
            for i in range(10):
                image_list.append(os.path.join(self.root, "uavid_{}".format(self.split), seq, "Images", "{:06d}.png".format(i*100)))
                label_list.append(os.path.join(self.root, "uavid_{}".format(self.split), seq, "Labels", "{:06d}.png".format(i*100)))

        self.images[self.split] = image_list
        self.labels[self.split] = label_list
        

    def __len__(self):
        return len(self.images[self.split])
    
    def __getitem__(self, index):
        img_path = self.images[self.split][index]
        lbl_path = self.labels[self.split][index]
        # Read image and label
        img = read_image(img_path)
        lbl = read_image(lbl_path).squeeze(0).long()

        # Resize
        img = TF.resize(img, (1024,2048), antialias=True)        
        lbl = TF.resize(lbl, (1024,2048), interpolation=TF.InterpolationMode.NEAREST, antialias=True)

        # RandomCrop
        i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.img_size[0], self.img_size[1]))
        img = TF.crop(img, i, j, h, w)
        lbl = TF.crop(lbl, i, j, h, w)


        # Encode labels
        lbl = self.encode_segmap(lbl)
        
        # Augmentations
        if self.split == "train":
            # Random flips
            if np.random.random() < 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            # Random color jitter
            if np.random.random() < 0.25:
                img = TF.adjust_brightness(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_contrast(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_saturation(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_hue(img, 0.10 * (np.random.random() - 0.5))

        # Normalize
        if self.img_norm:
            img = TF.normalize(img.float(), self.mean, self.std)

        return img, lbl
        

        

