import argparse
import itertools
import os.path
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NpyDataset
from network.unet import UNetGenerator, SpectralNormDiscriminator
from utils.losses import CycleLoss
from utils.utils import ReplayBuffer, LambdaLR, Logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--dataset_path', type=str, default='datasets_synthrad2025', help='root directory of the dataset')
    parser.add_argument('--dataset_name', type=str, default=None, help='dataset folder name under dataset_path')
    parser.add_argument('--anatomy', choices=['brain', 'pelvis', 'thorax'], default='thorax', help="The anatomy type")
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--model_path', type=str, default='checkpoint', help="Path to save model checkpoints")
    parser.add_argument('--experiment_name', type=str, default=None, help='checkpoint subfolder name')
    parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--cuda', dest='cuda', action='store_true', default=True, help='use GPU computation')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='use CPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from previous checkpoint')
    parser.add_argument('--unaligned', action='store_true', default=False, help='sample B independently from A')
    parser.add_argument('--augment', action='store_true', default=False, help='enable resize/crop/flip augmentation')
    parser.add_argument('--replay_buffer_size', type=int, default=50, help='number of generated samples to cache')
    parser.add_argument('--n_generator', type=int, default=2, help='generator update steps per batch')
    parser.add_argument('--max_batches', type=int, default=0, help='limit batches per epoch, 0 means no limit')
    parser.add_argument('--skip_save', action='store_true', default=False, help='skip checkpoint saving')
    parser.add_argument('--log', dest='log', action='store_true', default=True, help='enable Visdom logging')
    parser.add_argument('--no-log', dest='log', action='store_false', help='disable Visdom logging')
    parser.add_argument('--visdom_env', type=str, default=None, help='Visdom environment name')
    parser.add_argument('--visdom_plot_interval', type=int, default=10, help='iterations between Visdom loss updates')
    parser.add_argument('--visdom_image_interval', type=int, default=10, help='iterations between Visdom image updates')
    opt = parser.parse_args()
    print(opt)

    dataset_root = Path(opt.dataset_path)
    dataset_name = opt.dataset_name or f'{opt.anatomy}-{opt.size}'
    experiment_name = opt.experiment_name or f'{dataset_root.name}-{dataset_name}'

    opt.dataset_path = str(dataset_root / dataset_name)
    opt.model_path = str(Path(opt.model_path) / experiment_name)
    print(f"Dataset path: {opt.dataset_path}")
    print(f"Checkpoint path: {opt.model_path}")
    os.makedirs(opt.model_path, exist_ok=True)

    for split_domain in ('train/A', 'train/B'):
        data_dir = Path(opt.dataset_path) / split_domain
        if not data_dir.exists() or not list(data_dir.glob('*.npy')):
            raise FileNotFoundError(f"Training data not found: {data_dir}")

    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ###### Definition of variables ######
    # Networks
    netG_A2B = UNetGenerator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = UNetGenerator(opt.output_nc, opt.input_nc).to(device)
    netD_A = SpectralNormDiscriminator(opt.input_nc).to(device)
    netD_B = SpectralNormDiscriminator(opt.output_nc).to(device)

    # netG_A2B.apply(weights_init_normal)
    # netG_B2A.apply(weights_init_normal)
    # netD_A.apply(weights_init_normal)
    # netD_B.apply(weights_init_normal)

    if opt.resume:
        print('Resuming from previous checkpoint...')
        # Load state dicts
        netG_A2B.load_state_dict(
            torch.load(os.path.join(opt.model_path, 'netG_A2B.pth'), weights_only=False, map_location='cpu'))
        netG_B2A.load_state_dict(
            torch.load(os.path.join(opt.model_path, 'netG_B2A.pth'), weights_only=False, map_location='cpu'))
        netD_A.load_state_dict(
            torch.load(os.path.join(opt.model_path, 'netD_A.pth'), weights_only=False, map_location='cpu'))
        netD_B.load_state_dict(
            torch.load(os.path.join(opt.model_path, 'netD_B.pth'), weights_only=False, map_location='cpu'))

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    # criterion_cycle = torch.nn.L1Loss()
    criterion_cycle = CycleLoss(proportion_ssim=0.5)
    criterion_identity = torch.nn.L1Loss()

    lambda_GAN = 1.0  # GAN Loss 权重
    lambda_cycle = 10.0  # Cycle Loss 权重
    lambda_identity = 3.0  # Identity Loss 权重

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr * 4, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr * 4, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)

    input_A = torch.zeros(opt.batch_size, opt.input_nc, opt.size, opt.size, device=device, dtype=torch.float32)
    input_B = torch.zeros(opt.batch_size, opt.output_nc, opt.size, opt.size, device=device, dtype=torch.float32)

    target_size = int(opt.size // 8 - 2)
    target_real = torch.full((opt.batch_size, 1, target_size, target_size), 0.9, device=device, dtype=torch.float32)
    target_fake = torch.full((opt.batch_size, 1, target_size, target_size), 0.1, device=device, dtype=torch.float32)

    fake_A_buffer = ReplayBuffer(max_size=opt.replay_buffer_size)
    fake_B_buffer = ReplayBuffer(max_size=opt.replay_buffer_size)

    if opt.augment:
        transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BILINEAR),
                       transforms.RandomCrop(opt.size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])]
    else:
        transforms_ = [transforms.Resize((opt.size, opt.size), Image.BILINEAR),
                       transforms.ToTensor(),
                       transforms.Normalize([0.5], [0.5])]
    dataset = NpyDataset(opt.dataset_path, transforms_=transforms_, unaligned=opt.unaligned, anatomy=opt.anatomy)
    print(f"Training samples: {len(dataset)}")
    print(f"Paired sampling: {not opt.unaligned}")

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

    if opt.log:
        visdom_env = opt.visdom_env or experiment_name
        logger = Logger(opt.n_epochs, len(dataloader), env=visdom_env, image_size=opt.size,
                        image_interval=opt.visdom_image_interval, plot_interval=opt.visdom_plot_interval,
                        start_epoch=opt.epoch)

    for epoch in range(opt.epoch, opt.n_epochs + 1):
        data_loader_train = tqdm(dataloader, file=sys.stdout)
        train_losses = []

        for batch_idx, batch in enumerate(data_loader_train):
            # Set model input
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            input_A = real_A.detach().clone()
            input_B = real_B.detach().clone()

            ###### 训练判别器 A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = netG_B2A(real_B).detach()
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            ###### 训练判别器 B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = netG_A2B(real_A).detach()
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            ###### 让生成器多训练 opt.n_generator 次 ######
            for _ in range(opt.n_generator):
                optimizer_G.zero_grad()

                # Identity loss
                same_B = netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B) * lambda_identity
                same_A = netG_B2A(real_A)
                loss_identity_A = criterion_identity(same_A, real_A) * lambda_identity

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real) * lambda_GAN

                fake_A = netG_B2A(real_B)
                pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * lambda_GAN

                # Cycle loss
                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * lambda_cycle

                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * lambda_cycle

                # Total loss
                loss_G = (
                    loss_identity_A + loss_identity_B +
                    loss_GAN_A2B + loss_GAN_B2A +
                    loss_cycle_ABA + loss_cycle_BAB
                )
                loss_G.backward()
                optimizer_G.step()

                train_losses.append(loss_G.item())

            data_loader_train.desc = f"[train epoch {epoch}] loss: {np.mean(train_losses):.4f} "

            if opt.log:
                # Progress report (http://localhost:8097)
                logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                            'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                           images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

            if opt.max_batches > 0 and (batch_idx + 1) >= opt.max_batches:
                break

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        print(f"Generator LR: {lr_scheduler_G.get_last_lr()[0]:.6f}")
        print(f"Discriminator LR: {lr_scheduler_D_A.get_last_lr()[0]:.6f}")

        # Save models checkpoints
        if not opt.skip_save:
            torch.save(netG_A2B.state_dict(), os.path.join(opt.model_path, 'netG_A2B.pth'))
            torch.save(netG_B2A.state_dict(), os.path.join(opt.model_path, 'netG_B2A.pth'))
            torch.save(netD_A.state_dict(), os.path.join(opt.model_path, 'netD_A.pth'))
            torch.save(netD_B.state_dict(), os.path.join(opt.model_path, 'netD_B.pth'))
