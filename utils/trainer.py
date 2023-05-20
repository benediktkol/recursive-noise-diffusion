import logging
import math
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.evaluation import Evaluator, segmentation_cross_entropy, noise_mse, write_images_to_tensorboard
from utils.utils import diffuse, get_patch_indices, dynamic_range


class TrainerConfig:
    """
    Config settings (hyperparameters) for training.
    """
    # optimization parameters
    max_epochs = 100
    batch_size = 2
    learning_rate = 1e-5
    momentum = None
    weight_decay = 0.001 
    grad_norm_clip = 0.95

    # learning rate decay params
    lr_decay = True
    lr_decay_gamma = 0.98

    # network
    network = 'unet'

    # diffusion other settings
    train_on_n_scales = None
    not_recursive = False

    # checkpoint settings
    checkpoint_dir = 'output/checkpoints/'
    log_dir = 'output/logs/'
    load_checkpoint = None
    checkpoint = None
    weights_only = False

    # data
    dataset_selection = 'uavid'

    # other
    eval_every = 2
    save_every = 2
    seed = 0
    n_workers = 8

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def save_config_file(self, filename):
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        logging.info("Saving TrainerConfig file: {}".format(filename))
        with open(filename, 'w') as f:
            for k,v in vars(self).items():
                f.write("{}={}\n".format(k,v))

class Trainer:

    def __init__(self, model, network_config, config, train_data_loader, validation_data_loader=None):
        self.model = model
        self.network_config = network_config
        self.config = config
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.device = config.device

    def create_run_name(self):
        """Creates a unique run name based on current time and network"""
        self.run_name = '{}_{}'.format(time.strftime("%Y%m%d-%H%M"), self.config.network)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, id=None):
        """Saves a model checkpoint"""
        if id is None:
            id = "e{}".format(epoch)
        path = os.path.normpath(self.config.checkpoint_dir + "{}/{}_{}.pt".format(self.run_name, self.run_name, id)) # path/time_network/time_network_epoch.pt
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        logging.info("Saving checkpoint: {}".format(path))
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, path)

    def get_optimizer(self):
        """Defines the optimizer"""
        # optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.999), weight_decay=self.config.weight_decay)
        if (self.config.checkpoint is not None) and (self.config.weights_only is False):
            optimizer.load_state_dict(self.config.checkpoint['optimizer_state_dict'])
        return optimizer

    def get_scheduler(self, optimizer):
        """Defines the learning rate scheduler"""
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay_gamma)
        if (self.config.checkpoint is not None) and (self.config.weights_only is False):
            scheduler.load_state_dict(self.config.checkpoint['scheduler_state_dict'])
        return scheduler

    def denoise_loop_scales(self, model, network_config, config, images, seg_gt_one_hot, optimizer, scaler):
        """Denoises all scales for a single timestep"""
        # Calculate scale sizes (smallest first)
        scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

        # Initialize first prediction (random noise)
        seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

        # Denoise whole segmentation map in steps
        for timestep in range(network_config.n_timesteps): # for each step
            loss_per_scale = torch.zeros(network_config.n_scales)

            for scale in range(network_config.n_scales): # for each scale
                # break if we don't want to train on all scales
                if scale > config.train_on_n_scales - 1:
                    break
                # Resize to current scale
                images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
                seg_gt_scaled = F.interpolate(seg_gt_one_hot, size=scale_sizes[scale], mode='bilinear', align_corners=False)
                seg_previous_scaled = F.interpolate(seg_previous_scaled, size=scale_sizes[scale], mode='bilinear', align_corners=False)

                patch_indices = get_patch_indices(scale_sizes[scale], network_config.max_patch_size, overlap=False)

                # Create a new tensor to store the denoised segmentation map
                seg_denoised = torch.zeros(seg_previous_scaled.shape)
                # Create a tensor to store the number of times a pixel has been denoised
                n_denoised = torch.zeros(seg_previous_scaled.shape)

                for x, y, patch_size in patch_indices: # for each patch
                    # Get the patch
                    img_patch = images_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().cuda(non_blocking=True)
                    seg_gt_patch = seg_gt_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().cuda(non_blocking=True)
                    seg_patch_previous = seg_previous_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().cuda(non_blocking=True).softmax(dim=1)
                    if config.not_recursive:
                        if timestep + scale > 0:
                            seg_patch_previous = seg_gt_patch

                    # Diffuse
                    t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]).cuda(non_blocking=True) # time step
                    seg_patch_diffused = diffuse(seg_patch_previous, t).detach() # diffuse segmentation map
                    noise_gt = seg_patch_diffused - seg_gt_patch # The noise added in the diffusion process + the error from the previous step

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Runs the forward pass with autocasting
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        noise_predicted = model(seg_patch_diffused, img_patch, t) # predict the noise
                        seg_patch_denoised = seg_patch_diffused - noise_predicted # denoise the patch

                        # Compute loss
                        losses = {}
                        noise_mse_loss = noise_mse(noise_predicted, noise_gt)
                        losses['noise_mse'] = noise_mse_loss
                        # seg_cross_entropy_loss = segmentation_cross_entropy(seg_patch_denoised, seg_gt_patch.argmax(dim=1))
                        # losses['seg_cross_entropy'] = seg_cross_entropy_loss
                        total_loss = noise_mse_loss

                    # Backward pass
                    # total_loss.backward()
                    scaler.scale(total_loss).backward()
                    
                    # Clip the gradients
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

                    # Update the parameters
                    # optimizer.step() 
                    scaler.step(optimizer)

                    # Update the scale for the next iteration.
                    scaler.update()

                    # Add the denoised patch to the segmentation map
                    seg_patch_denoised = seg_patch_denoised.detach().cpu() # detach from the graph
                    seg_denoised[:, :, x:x+patch_size, y:y+patch_size] += seg_patch_denoised
                    n_denoised[:, :, x:x+patch_size, y:y+patch_size] += 1 

                
                # Average the denoised patches
                seg_denoised = seg_denoised / n_denoised

                # # Adjust range
                # seg_denoised = dynamic_range(seg_denoised)

                # Update the previous segmentation map
                seg_previous_scaled = seg_denoised
        
        return seg_denoised, losses
    
    def denoise_linear_scales(self, model, network_config, config, images, seg_gt_one_hot, optimizer, scaler):
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
                seg_gt_scaled = F.interpolate(seg_gt_one_hot, size=scale_sizes[scale], mode='nearest')
                seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False)

                patch_indices = get_patch_indices(scale_sizes[scale], network_config.max_patch_size, overlap=False)

            # Create a new tensor to store the denoised segmentation map
            seg_denoised = torch.zeros(seg_previous_scaled.shape)
            # Create a tensor to store the number of times a pixel has been denoised
            n_denoised = torch.zeros(seg_previous_scaled.shape)

            for x, y, patch_size in patch_indices: # for each patch
                # Get the patch
                img_patch = images_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(self.device).contiguous()
                seg_gt_patch = seg_gt_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(self.device).contiguous()
                seg_patch_previous = seg_previous_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(self.device).contiguous()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Diffuse
                t = torch.tensor([(network_config.n_timesteps - timestep) / network_config.n_timesteps]).to(self.device) # time step
                seg_patch_diffused = diffuse(seg_patch_previous, t).detach() # diffuse segmentation map
                noise_gt = seg_patch_diffused - seg_gt_patch # The noise added in the diffusion process + the error from the previous step

                # Forward pass
                noise_predicted = model(seg_patch_diffused, img_patch, t) # predict the noise
                seg_patch_denoised = seg_patch_diffused - noise_predicted # denoise the patch

                # Compute loss
                losses = {}
                noise_mse_loss = noise_mse(noise_predicted, noise_gt)
                losses['noise_mse'] = noise_mse_loss
                seg_cross_entropy_loss = segmentation_cross_entropy(seg_patch_denoised, seg_gt_patch.argmax(dim=1))
                losses['seg_cross_entropy'] = seg_cross_entropy_loss
                total_loss = noise_mse_loss + seg_cross_entropy_loss

                # Backward pass
                total_loss.backward()
                
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

                # Update the parameters
                optimizer.step() 

                # Add the denoised patch to the segmentation map
                seg_patch_denoised = seg_patch_denoised.detach().cpu() # detach from the graph
                seg_denoised[:, :, x:x+patch_size, y:y+patch_size] += seg_patch_denoised
                n_denoised[:, :, x:x+patch_size, y:y+patch_size] += 1 

            # Average the denoised patches
            seg_denoised = seg_denoised / n_denoised

            # Update the previous segmentation map
            seg_previous_scaled = seg_denoised

        return seg_denoised, losses
    
    def denoise_and_backprop(self, model, network_config, config, images, seg_gt_one_hot, optimizer, scaler):
        """Denoises and backpropagates the error"""
        if network_config.scale_procedure == 'loop':
            seg_denoised, losses = self.denoise_loop_scales(model, network_config, config, images, seg_gt_one_hot, optimizer, scaler)
        elif network_config.scale_procedure == 'linear':
            seg_denoised, losses = self.denoise_linear_scales(model, network_config, config, images, seg_gt_one_hot, optimizer, scaler)

        return seg_denoised, losses

    def train(self):
        """Trains the model"""
        self.create_run_name()
        model = self.model
        network_config = self.network_config
        config = self.config
        optimizer = self.get_optimizer()
        scaler = GradScaler()
        scheduler = self.get_scheduler(optimizer)
        writer = SummaryWriter(log_dir=(config.log_dir + self.run_name))
        evaluator = Evaluator(model, network_config, self.device, dataset_selection=config.dataset_selection, validation_data_loader=self.validation_data_loader, writer=writer)

        config.save_config_file(os.path.normpath(config.checkpoint_dir + "{}/{}_config.txt".format(self.run_name, self.run_name)))

        def run_epoch():
            model.train()
            
            pbar_epoch = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader), desc='Epoch {}/{}'.format(epoch+1, config.max_epochs), leave=False, bar_format='{l_bar}{bar:50}{r_bar}')
            for it, samples in pbar_epoch:
                # Unpack the samples
                images, seg_gt = samples
                seg_gt_one_hot = F.one_hot(seg_gt, num_classes=network_config.n_classes+1).permute(0,3,1,2)[:,:-1,:,:].float() # make one hot (if remove void class [:,:-1,:,:])

                # Denoise and backpropagate
                seg_denoised, losses = self.denoise_and_backprop(model, network_config, config, images, seg_gt_one_hot, optimizer, scaler)
                
                # Write to tensorboard
                it_total = it + epoch*len(self.train_data_loader)
                if it_total % 10 == 0 and it_total > 0:
                    for loss_name, loss in losses.items():
                        writer.add_scalar('train/{}'.format(loss_name), loss, it_total)
                
                # Write images to tensorboard
                if it % 200 == 0:
                    write_images_to_tensorboard(writer, it_total, image=images[0], seg_predicted=seg_denoised[0], seg_gt=seg_gt[0], datasplit='train', dataset_name=config.dataset_selection)
            
            scheduler.step()
            

        with logging_redirect_tqdm():
            pbar_total = tqdm(range(config.max_epochs), desc='Total', bar_format='{l_bar}{bar:50}{r_bar}')
            for epoch in pbar_total:
                # Run an epoch
                run_epoch()

                # Save checkpoint
                if (epoch+1) % config.save_every == 0:
                    self.save_checkpoint(model, optimizer, scheduler, epoch+1)
                
                # Evaluate
                if self.validation_data_loader is not None:
                    if (epoch+1) % config.eval_every == 0:
                        evaluator.validate(epoch+1)
            
            writer.flush()
            writer.close()