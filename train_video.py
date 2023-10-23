import argparse
import utils
import random
import os
import numpy as np
import cv2


from utils import logger, tools
import logging
import colorama

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.video import SingleVideoDataset
from datasets.generate_frames import video_to_frames 

from modules import networks_3d
from modules.losses import kl_criterion, reconstruction_loss, adversarial_loss, PerceptualLoss
from modules.utils import *


clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT

# Specify the path where you want to create the folder
output = 'outputs'  # Replace with the desired path

def train(opt, netG):
    # Re-generate dataset frames
    fps, td, fps_index = utils.get_fps_td_by_index(opt.scale_idx, opt)
    opt.fps = fps
    opt.td = td
    opt.fps_index = fps_index

    with logger.LoggingBlock("Updating dataset", emph=True):
        logging.info("{}FPS :{} {}{}".format(green, clear, opt.fps, clear))
        logging.info("{}Time-Depth :{} {}{}".format(green, clear, opt.td, clear))
        logging.info("{}Sampling-Ratio :{} {}{}".format(green, clear, opt.sampling_rates[opt.fps_index], clear))
        opt.dataset.generate_frames(opt.scale_idx)

    # Initialize noise
    if not hasattr(opt, 'Z_init_size'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        opt.ar = 1.0  # Set a default value for opt.ar
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, opt.td, *initial_size]

    if opt.vae_levels < opt.scale_idx + 1:
        D_curr = getattr(networks_3d, opt.discriminator)(opt).to(opt.device)

        if (opt.netG != '') and (opt.resumed_idx == opt.scale_idx):
            D_curr.load_state_dict(
                torch.load('{}/netD_{}.pth'.format(opt.resume_dir, opt.scale_idx - 1))['state_dict'])
        elif opt.vae_levels < opt.scale_idx:
            D_curr.load_state_dict(
                torch.load('{}/netD_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict'])

        # Current optimizers
        optimizerD = optim.Adam(D_curr.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    parameter_list = []
    # Generator Adversary
    if not opt.train_all:
        if opt.vae_levels < opt.scale_idx + 1:
            train_depth = min(opt.train_depth, len(netG.body) - opt.vae_levels + 1)
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-train_depth:])]
        else:
            # VAE
            parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]
    else:
        if len(netG.body) < opt.train_depth:
            parameter_list += [{"params": netG.encode.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)},
                               {"params": netG.decoder.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body) - 1 - idx))}
                for idx, block in enumerate(netG.body)]
        else:
            parameter_list += [
                {"params": block.parameters(),
                 "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(netG.body[-opt.train_depth:])]

    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Parallel
    if opt.device == 'cuda':
        G_curr = torch.nn.DataParallel(netG)
        if opt.vae_levels < opt.scale_idx + 1:
            D_curr = torch.nn.DataParallel(D_curr)
    else:
        G_curr = netG

    progressbar_args = {
        "iterable": range(opt.niter),
        "desc": "Training scale [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1),
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    epoch_iterator = tools.create_progressbar(**progressbar_args)

    iterator = iter(data_loader)

    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        if opt.scale_idx > 0:
            real, real_zero = data
            real = real.to(opt.device)
            real_zero = real_zero.to(opt.device)
        else:
            real = data.to(opt.device)
            real_zero = real

        noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

        ############################
        # calculate noise_amp
        ###########################
        if iteration == 0:
            if opt.const_amp:
                opt.Noise_Amps.append(1)
            else:
                with torch.no_grad():
                    if opt.scale_idx == 0:
                        opt.noise_amp = 1
                        opt.Noise_Amps.append(opt.noise_amp)
                    else:
                        opt.Noise_Amps.append(0)
                        z_reconstruction, _, _ = G_curr(real_zero, opt.Noise_Amps, mode="rec")

                        RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                        opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                        opt.Noise_Amps[-1] = opt.noise_amp

        ############################
        # (1) Update VAE network
        ###########################
        total_loss = 0

        generated, generated_vae, (mu, logvar) = G_curr(real_zero, opt.Noise_Amps, mode="rec")

        if opt.vae_levels >= opt.scale_idx + 1:
            rec_vae_loss = opt.rec_loss(generated, real) + opt.rec_loss(generated_vae, real_zero)
            kl_loss = kl_criterion(mu, logvar)
            '''vae_loss = opt.rec_weight * rec_vae_loss + opt.kl_weight * kl_loss'''
            rec_weight = 50  # Adjust the weight as needed
            vae_loss = rec_weight * full_rec_loss + kl_loss

            total_loss += vae_loss
        else:
            ############################
            # (2) Update D network: maximize D(x) + D(G(z))
            ###########################
            # train with real
            #################

            # Train 3D Discriminator
            D_curr.zero_grad()
            output = D_curr(real)
            errD_real = -output.mean()

            # train with fake
            #################
            fake, _ = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, mode="rand")

            # Train 3D Discriminator
            output = D_curr(fake.detach())
            errD_fake = output.mean()

            gradient_penalty = calc_gradient_penalty(D_curr, real, fake, opt.lambda_grad, opt.device)
            errD_total = errD_real + errD_fake + gradient_penalty
            errD_total.backward()
            optimizerD.step()

            ############################
            # (3) Update G network: maximize D(G(z))
            ###########################
            errG_total = 0
            rec_loss = reconstruction_loss(generated, real)  # Assuming generated frames are the reconstructed output
            rec_loss1 = reconstruction_loss(generated_vae, real)  # Assuming generated_vae frames are the differently reconstructed output
            full_rec_loss = rec_loss + rec_loss1

            # Train with 3D Discriminator
            output = D_curr(fake)
            errG = -output.mean() * opt.disc_loss_weight
            adversarial_loss(generated, target_is_real=True) 
            errG_total += adversarial_loss(output, target_is_real=True)
            total_loss += errG_total

        G_curr.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(G_curr.parameters(), opt.grad_clip)
        optimizerG.step()

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        if opt.visualize:
            # Tensorboard
            opt.summary.add_scalar('Video/Scale {}/rec vae loss'.format(opt.scale_idx), rec_vae_loss.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/vae loss'.format(opt.scale_idx), vae_loss.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/adversarial loss'.format(opt.scale_idx), errG.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/gradient penalty'.format(opt.scale_idx), gradient_penalty.item(), iteration)
            '''opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
            if opt.vae_levels >= opt.scale_idx + 1:
                opt.summary.add_scalar('Video/Scale {}/KLD'.format(opt.scale_idx), kl_loss.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/rec loss'.format(opt.scale_idx), rec_loss.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)
            if opt.vae_levels < opt.scale_idx + 1:
                opt.summary.add_scalar('Video/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/errD_real'.format(opt.scale_idx), errD_real.item(), iteration)
            else:
                opt.summary.add_scalar('Video/Scale {}/Rec VAE'.format(opt.scale_idx), rec_vae_loss.item(), iteration)
'''

            
            if opt.visualize and iteration % opt.print_interval == 0:
                with torch.no_grad():
                    fake_var = []
                    fake_vae_var = []
                    for _ in range(3):
                        noise_init = utils.generate_noise(ref=noise_init)
                        fake, fake_vae = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, mode="rand")
                        fake_var.append(fake)
                        fake_vae_var.append(fake_vae)
                    fake_var = torch.cat(fake_var, dim=0)
                    fake_vae_var = torch.cat(fake_vae_var, dim=0)

                opt.summary.visualize_video(opt, iteration, real, 'Real')
                opt.summary.visualize_video(opt, iteration, generated, 'Generated')
                opt.summary.visualize_video(opt, iteration, generated_vae, 'Generated VAE')
                opt.summary.visualize_video(opt, iteration, fake_var, 'Fake var')
                opt.summary.visualize_video(opt, iteration, fake_vae_var, 'Fake VAE var')



    epoch_iterator.close()

    # Save data
    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': netG.state_dict(),
        'optimizer': optimizerG.state_dict(),
        'noise_amps': opt.Noise_Amps,
    }, 'netG.pth')
    if opt.vae_levels < opt.scale_idx + 1:
        opt.saver.save_checkpoint({
            'scale': opt.scale_idx,
            'state_dict': D_curr.module.state_dict() if opt.device == 'cuda' else D_curr.state_dict(),
            'optimizer': optimizerD.state_dict(),
        }, 'netD_{}.pth'.format(opt.scale_idx))

def create_dataset(frames_directory, transforms=None, subset_pct=0.2, num_sparse_frames=None):
    frames = []
    for filename in os.listdir(frames_directory):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(frames_directory, filename)
            frame = cv2.imread(image_path)
            frames.append(frame)

    dataset = SingleVideoDataset(frames, transforms=transforms, subset_pct=subset_pct, num_sparse_frames=num_sparse_frames)
    return dataset

frames_directory = 'framess'  # Replace with the path to your frames directory
dataset = create_dataset(frames_directory, transforms=None, subset_pct=0.2, num_sparse_frames=None)