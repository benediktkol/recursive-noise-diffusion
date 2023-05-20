import logging
import numpy as np
import os
import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils import data

def decode_segmap(seg, is_one_hot=False):
        colors = torch.tensor([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0]
        ], dtype=torch.uint8)
        if is_one_hot:
            seg = torch.argmax(seg, dim=0)
        # convert classes to colors
        seg_img = torch.empty((seg.shape[0], seg.shape[1], 3), dtype=torch.uint8)
        for c in range(20):
            seg_img[seg == c, :] = colors[c]
        return seg_img.permute(2, 0, 1)


class CityscapesLoader(data.Dataset):
    """CityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    def recursive_glob(self, rootdir=".", suffix=""):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]


    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]
    ]

    label_colours = dict(zip(range(19), colors))


    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(1024, 2048),
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.test_mode = test_mode
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.ignore_index = 255
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
            self.ignore_index,
        ]
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "unlabelled",
        ]

        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        logging.info('Found {} {} images'.format(len(self.files[split]), split))
    
    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        # Read image and label
        img = read_image(img_path)
        lbl = read_image(lbl_path).squeeze(0).long()

        lbl = self.encode_segmap(lbl)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """

        # # Random crop
        # if self.split == "train":
        #     i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.img_size[0], self.img_size[1]))
        #     img = TF.crop(img, i, j, h, w)
        #     lbl = TF.crop(lbl, i, j, h, w)

        # Random horizontal flipping
        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            # Random color jitter
            if np.random.random() < 0.25:
                img = TF.adjust_brightness(img, 0.5 + np.random.random())
                img = TF.adjust_contrast(img, 0.5 + np.random.random())
                img = TF.adjust_saturation(img, 0.5 + np.random.random())
                img = TF.adjust_hue(img, 0.10 * (np.random.random() - 0.5))
            # Random resized crop
            if np.random.random() < 0.25:
                i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.5, 1), ratio=(2, 2), antialias=True)
                img = TF.resized_crop(img, i, j, h, w, self.img_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
                lbl = TF.resized_crop(lbl.unsqueeze(0), i, j, h, w, self.img_size, interpolation=transforms.InterpolationMode.NEAREST, antialias=True).squeeze()

                
        
        # Normalize
        if self.img_norm:
            img = TF.normalize(img.float(), self.mean, self.std)


        return img, lbl


    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

#     local_path = "/datasets01/cityscapes/112817/"
#     dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
#     bs = 4
#     trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
#     for i, data_samples in enumerate(trainloader):
#         imgs, labels = data_samples
#         import pdb

#         pdb.set_trace()
#         imgs = imgs.numpy()[:, ::-1, :, :]
#         imgs = np.transpose(imgs, [0, 2, 3, 1])
#         f, axarr = plt.subplots(bs, 2)
#         for j in range(bs):
#             axarr[j][0].imshow(imgs[j])
#             axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
#         plt.show()
#         a = input()
#         if a == "ex":
#             break
#         else:
#             plt.close()