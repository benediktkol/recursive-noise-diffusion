"""
Assortment of useful methods
"""

import logging
import numpy as np
import random
import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info('This run is deterministic with seed {}'.format(seed))

def diffuse(seg_start, t, noise_type='normal_add', schedule='linear'):
    """Diffuse a segmentation map by adding noise based on the diffusion schedule"""
    # Define the diffusion schedule
    if schedule == 'linear':
        noise_factor = t
    elif schedule == 'square':
        noise_factor = t ** 2

    noise_factor = noise_factor.view(-1, 1, 1, 1)

    # Create noise
    if noise_type == 'normal_add':
        noise = torch.randn(seg_start.shape, device=seg_start.device)
        diffused = seg_start + noise * noise_factor
    elif noise_type == 'normal_average':
        noise = torch.randn(seg_start.shape, device=seg_start.device)
        diffused = (1 - noise_factor) * (seg_start - 0.5) + 0.5 + noise_factor * noise
    elif noise_type == 'uniform':
        noise = torch.rand(seg_start.shape, device=seg_start.device)
        diffused = seg_start + noise * noise_factor
    elif noise_type == 'binary':
        noise = F.one_hot(torch.randint(0, seg_start.shape[1], (seg_start.shape[0], seg_start.shape[2], seg_start.shape[3]), device=seg_start.device), num_classes=seg_start.shape[1]).permute(0,3,1,2).float()
        noise_factor = noise_factor.expand(-1, 1, seg_start.shape[2], seg_start.shape[3])
        noise_factor_bernoulli = torch.bernoulli(noise_factor)
        diffused = seg_start * (1 - noise_factor_bernoulli) + noise * noise_factor_bernoulli
    elif noise_type == 'none':
        diffused = seg_start

    # Clip the values
    diffused = torch.clamp(diffused, 0, 1)

    return diffused

def get_patch_indices(img_size, patch_size, overlap=True):
    """
    Get the indices of the patches in an image
    """
    # Adjust patch size if necessary
    # if min(img_size[0], img_size[1]) < patch_size:
    #     patch_size = min(img_size[0], img_size[1])

    # Set stride
    if overlap:
        stride = patch_size // 2
    else:
        stride = patch_size

    # Get the number of patches
    n_patches_h = max((img_size[0] - patch_size) // stride + 1, 1)
    n_patches_w = max((img_size[1] - patch_size) // stride + 1, 1)

    # Get the indices of the patches
    patch_indices = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            x = i * stride
            y = j * stride
            patch_indices.append((x, y, patch_size))

    return patch_indices

def dynamic_range(x, mode='argmax'):
    """
    Adjust the dynamic range of a tensor
    """
    if mode == 'softmax':
        x = torch.softmax(x, dim=1)
    elif mode == 'argmax':
        v,_ = x.topk(1, dim=1)
        x = x/v
        x = x.trunc()
    elif mode == 'sigmoid':
        x = torch.sigmoid(x)
    elif mode == 'clamp':
        x = torch.clamp(x, 0, 1)
    elif mode == 'dynamic':
        s = max(x.max(), 1)
        x = torch.clamp(x, min=-s, max=s)
        x = x / s
    return x

def denoise_scale(model, device, seg_diffused, images, t, patch_size=512, overlap=False, use_dynamic_range=False):
    """
    Denoise a segmentation map using a model
    """
    # Get the indices of the patches
    img_size = seg_diffused.shape[2:]
    patch_indices = get_patch_indices(img_size, patch_size, overlap)

    # Create a new tensor to store the denoised segmentation map
    seg_denoised = torch.zeros(seg_diffused.shape)
    # Create a tensor to store the number of times a pixel has been denoised
    n_denoised = torch.zeros(seg_diffused.shape)

    # Denoise each patch
    for x, y, patch_size in patch_indices:
        # Get the patch
        img_patch = images[:, :, x:x+patch_size, y:y+patch_size].detach().to(device).contiguous()
        seg_patch_diffused = seg_diffused[:, :, x:x+patch_size, y:y+patch_size].detach().to(device).contiguous()

        # Denoise the patch
        noise_predicted = model(seg_patch_diffused, img_patch, t.to(device)) # predict the noise
        seg_patch_denoised = seg_patch_diffused - noise_predicted # denoise the patch

        # Add the denoised patch to the segmentation map
        seg_denoised[:, :, x:x+patch_size, y:y+patch_size] += seg_patch_denoised.cpu()
        n_denoised[:, :, x:x+patch_size, y:y+patch_size] += 1
    
    # Average the denoised segmentation map
    seg_denoised /= n_denoised

    # Adjust range
    if use_dynamic_range:
        seg_denoised = dynamic_range(seg_denoised)

    return seg_denoised
