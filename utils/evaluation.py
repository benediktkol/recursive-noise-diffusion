"""Evaluates the performance of a model"""
import logging
import math
import torch
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import JaccardIndex, F1Score
from tqdm import tqdm

from utils.cityscapes_loader import decode_segmap as decode_segmap_cityscapes
from utils.utils import diffuse, denoise_scale
from utils.uavid_loader import decode_segmap as decode_segmap_uavid
from utils.vaihingen_buildings_loader import decode_segmap as decode_segmap_vaihingen

def segmentation_cross_entropy(predicted_segmentation, target_segmentation):
    """Returns Cross Entropy Loss"""
    weights = torch.tensor([1.79, 1.0, 2.17, 1.17, 3.2, 27.14, 21.56, 190.25], dtype=torch.float32).to(target_segmentation.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss = criterion(predicted_segmentation, target_segmentation)
    return loss

def noise_mse(noise_predicted, noise_target):
    """Returns MSE Loss"""
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(noise_predicted, noise_target)
    return loss

def compute_total_loss(segmentation_cross_entropy):
    """Returns total loss"""
    total_loss =  (1 * segmentation_cross_entropy)
    return total_loss

def write_images_to_tensorboard(writer, epoch, image=None, seg_diffused=None, seg_predicted=None, seg_gt=None, datasplit='validation', dataset_name='cityscapes'):
        """Writes images to TensorBoard"""
        # decode segmap based on dataset
        if dataset_name == 'cityscapes':
            decode_segmap = decode_segmap_cityscapes
        elif dataset_name == 'uavid':
            decode_segmap = decode_segmap_uavid
        elif dataset_name == 'vaihingen':
            decode_segmap = decode_segmap_vaihingen
        else:
            raise NotImplementedError('Dataset {} not implemented'.format(dataset_name))
        if image is not None:
            image = torchvision.utils.make_grid(image, normalize=True) # normalize to [0,1] and convert to uint8
            writer.add_images('{}/image'.format(datasplit), image, epoch, dataformats='CHW')
        if seg_diffused is not None:
            seg_diffused = decode_segmap(seg_diffused, is_one_hot=True)
            writer.add_images('{}/seg_diffused'.format(datasplit), seg_diffused, epoch, dataformats='CHW')
        if seg_predicted is not None:
            seg_predicted = decode_segmap(seg_predicted, is_one_hot=True)
            writer.add_images('{}/seg_predicted'.format(datasplit), seg_predicted, epoch, dataformats='CHW')
        if seg_gt is not None:
            seg_gt = decode_segmap(seg_gt, is_one_hot=False)
            writer.add_images('{}/seg_gt'.format(datasplit), seg_gt, epoch, dataformats='CHW')

def denoise_loop_scales(model, device, network_config, images):
    """Denoises all scales for a single timestep"""
    # Calculate scale sizes (smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # Initialize first prediction (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Initialize built in ensemble
    seg_denoised_ensemble = torch.zeros(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        
        for scale in range(network_config.n_scales): # for each scale
            # Resize to current scale
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False).softmax(dim=1)

            # Diffuse
            t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step
            seg_diffused = diffuse(seg_previous_scaled, t)  
            # Denoise
            seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

            # Update the previous segmentation map
            seg_previous_scaled = seg_denoised
        
        # Add to ensemble
        if network_config.built_in_ensemble:
            if timestep == 0:
                seg_denoised_ensemble = seg_denoised
            else:
                seg_denoised_ensemble = seg_denoised_ensemble / 2 + seg_denoised / 2
            
            seg_previous_scaled = seg_denoised_ensemble
    
    return seg_denoised

def denoise_linear_scales(model, device, network_config, images):
    """Denoises one scale at a each timestep"""
    # Calculate scale sizes (smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # Initialize first prediction (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        # Get the current scale
        timesteps_per_scale = math.ceil(network_config.n_timesteps / network_config.n_scales)
        scale = timestep // timesteps_per_scale
        
        # Resize to current scale
        if timestep % timesteps_per_scale == 0:
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False)

        # Diffuse
        t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step
        seg_diffused = diffuse(seg_previous_scaled, t)
        # Denoise
        seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

        # Update the previous segmentation map
        seg_previous_scaled = seg_denoised

    return seg_denoised

def denoise(model, device, network_config, images):
        """Denoises the segmentation map"""
        if network_config.scale_procedure == 'loop':
            seg_denoised = denoise_loop_scales(model, device, network_config, images)
        elif network_config.scale_procedure == 'linear':
            seg_denoised = denoise_linear_scales(model, device, network_config, images)

        return seg_denoised

class Evaluator:
    """Evaluates the performance of a model"""
    def __init__(self, model, network_config, device, dataset_selection=None, test_data_loader=None, validation_data_loader=None, writer=None):
        self.model = model
        self.network_config = network_config
        self.device = device
        self.dataset_selection = dataset_selection
        self.test_data_loader = test_data_loader
        self.validation_data_loader = validation_data_loader
        self.writer = writer

    def evaluate(self, data_loader, epoch=1, is_test=True, ensemble=1): # epoch=None
        """Evaluates the model on the given dataset"""
        model = self.model
        network_config = self.network_config
        model.eval()

        if self.dataset_selection == 'cityscapes':
            ignore_index = 19
            n_ignore = 1
        else:
            ignore_index = None
            n_ignore = 0

        jaccard_index = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index)
        jaccard_per_class = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index, average='none')
        f1_score = F1Score(num_classes=data_loader.dataset.n_classes + n_ignore, mdmc_average='samplewise')

        with torch.no_grad():
            pbar_eval = tqdm(enumerate(data_loader), total=len(data_loader), desc='{}'.format('Test' if is_test else 'Validation'), leave=is_test, bar_format='{l_bar}{bar:50}{r_bar}')
            for it, samples in pbar_eval:
                # Unpack the samples
                images, seg_gt = samples

                seg_denoised = denoise(model, self.device, network_config, images)

                # Ensamble
                for i in range(ensemble-1):
                    seg_denoised += denoise(model, self.device, network_config, images)
                seg_denoised /= ensemble

                # Compute loss
                seg_predicted = seg_denoised.view(seg_denoised.shape[0], seg_denoised.shape[1], -1).argmax(dim=1)
                seg_target = seg_gt.view(seg_gt.shape[0], -1)
                jaccard_index.update(seg_predicted, seg_target)
                jaccard_per_class.update(seg_predicted, seg_target)
                f1_score.update(seg_predicted, seg_target)

                # Write images to tensorboard
                if self.writer is not None:
                    if it < 8: 
                        write_images_to_tensorboard(self.writer, epoch, image=images[0], seg_predicted=seg_denoised[0], seg_gt=seg_gt[0], datasplit='validation/{}'.format(it))
                
            
        # Overall metrics
        jaccard_index_total = jaccard_index.compute()
        jaccard_per_class_total = jaccard_per_class.compute()
        f1_score_total = f1_score.compute()

        # Text report
        report = 'Jaccard index: {:.4f} | F1 score: {:.4f}'.format(jaccard_index_total, f1_score_total)
        report_per_class = 'Jaccard index per class: {}'.format(jaccard_per_class_total)
        if self.writer is None:
            logging.log(logging.WARNING, report)
            logging.log(logging.WARNING, report_per_class)
        else:
            logging.info('{} | {} | {}'.format("Test" if is_test else "Validation | Epoch: {}".format(epoch), report, report_per_class))

        # Write to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('{}/JaccardIndex'.format('test' if is_test else 'validation'), jaccard_index_total, epoch)
            self.writer.add_scalar('{}/F1Score'.format('test' if is_test else 'validation'), f1_score_total, epoch)

        
    def validate(self, epoch):
        """Evaluates the model on the validation dataset"""
        self.evaluate(self.validation_data_loader, epoch, is_test=False)

    def test(self, ensemble=1):
        """Evaluates the model on the test dataset"""
        self.evaluate(self.test_data_loader, is_test=True, ensemble=ensemble)


