# coding=utf-8
# Copyright (c) ganslate Contributors
# Changes added by Maastro-CDS-Imaging-Group : https://github.com/Maastro-CDS-Imaging-Group/ganslate
# Clean and simplify SSIM computation similar to fastMRI SSIM.

# Taken from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
# Licensed under MIT.
# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
# Some changes are made to work together with DIRECT.

# ----------------------------------------------------
# Taken from DIRECT https://github.com/directgroup/direct
# Copyright (c) DIRECT Contributors
# Added support for mixed precision by allowing one image to be of type `half` and the other `float`.
# ----------------------------------------------------

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma, device=None, dtype=None):
    """
    Create a 1D gaussian kernel
    Parameters
    ----------
    size : int
        The size of the gaussian kernel
    sigma : float
        The standard deviation of the normal distribution
    Returns
    -------
    torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=dtype, device=device).float()
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    # Return window as 1x1xsize
    return g.view(1, 1, *g.shape)


def gaussian_filter(input, win):
    """
    Blur input with 1D kernel
    """
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=input.shape[1])


class SSIMLoss(torch.nn.Module):

    def __init__(self, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """
        super().__init__()
        self.win_size, self.win_sigma = win_size, win_sigma
        self.K = K

    def forward(self, X, Y, data_range=1):
        assert X.shape == Y.shape, "X and Y need to be the same shape"
        assert X.ndim in [4, 5], "Dimensions of input must be NxCxHxW or NxCxDxHxW"

        # if NxCxDxHxW, convert NxC to N only giving NxDxHxW
        if X.ndim == 5:
            X = X.view(-1, *X.shape[2:])
            Y = Y.view(-1, *Y.shape[2:])
        channels = X.shape[1]

        # Create 1D gaussian window and repeat it over channel dims
        win = _fspecial_gauss_1d(self.win_size, self.win_sigma, dtype=X.dtype, device=X.device)
        win = win.repeat(channels, 1, 1, 1)

        K1, K2 = self.K
        compensation = 1.0

        C1 = (K1 * data_range)**2
        C2 = (K2 * data_range)**2

        mu1 = gaussian_filter(X, win)
        mu2 = gaussian_filter(Y, win)

        sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1.pow(2))
        sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2.pow(2))
        sigma12 = compensation * (gaussian_filter(X * Y, win) - (mu1 * mu2))

        S1 = (2 * mu1 * mu2 + C1) / (mu1.pow(2) + mu2.pow(2) + C1)
        S2 = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1

        # SSIM Distance metric approximation from: https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf
        # Add relu here since floating point rounding errors can lead this value to be slightly negative!
        S = torch.relu(2 - (S1 + S2))
        D_map = torch.sqrt(S)
        return D_map.mean()
class CycleGANLosses:
    """Defines losses used for optiming the generators in CycleGAN setup.
    Consists of:
        (1) Cycle-consistency loss (weighted combination of L1 and, optionally, SSIM)
        (2) Identity loss
    """

    def __init__(self, conf):
        self.lambda_AB = conf.train.gan.optimizer.lambda_AB
        self.lambda_BA = conf.train.gan.optimizer.lambda_BA

        lambda_identity = conf.train.gan.optimizer.lambda_identity
        proportion_ssim = conf.train.gan.optimizer.proportion_ssim

        # Cycle-consistency - L1, with optional weighted combination with SSIM
        self.criterion_cycle = CycleLoss(proportion_ssim)
        if lambda_identity > 0:
            self.criterion_idt = IdentityLoss(lambda_identity)
        else:
            self.criterion_idt = None

    def is_using_identity(self):
        """Check if idt_A and idt_B should be computed."""
        return True if self.criterion_idt else False

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']
        rec_A, rec_B = visuals['rec_A'], visuals['rec_B']
        idt_A, idt_B = visuals['idt_A'], visuals['idt_B']

        losses = {}

        # cycle-consistency loss
        # || G_BA(G_AB(real_A)) - real_A||
        losses['cycle_A'] = self.lambda_AB * self.criterion_cycle(real_A, rec_A)
        # || G_AB(G_BA(real_B)) - real_B||
        losses['cycle_B'] = self.lambda_BA * self.criterion_cycle(real_B, rec_B)

        # identity loss
        if self.criterion_idt:
            if idt_A is not None and idt_B is not None:
                # || G_AB(real_B) - real_B ||
                losses['idt_B'] = self.lambda_AB * self.criterion_idt(idt_B, real_B)
                # || G_BA(real_A) - real_A ||
                losses['idt_A'] = self.lambda_BA * self.criterion_idt(idt_A, real_A)

            else:
                raise ValueError(
                    "idt_A and/or idt_B is not computed but the identity loss is defined.")

        return losses


class CycleLoss:
    def __init__(self, proportion_ssim):
        self.criterion = torch.nn.L1Loss()
        if proportion_ssim > 0:
            self.ssim_criterion = SSIMLoss()
            # weights for addition of SSIM and L1 losses
            self.alpha = proportion_ssim
            self.beta = 1 - proportion_ssim
        else:
            self.ssim_criterion = None

    def __call__(self, real, reconstructed):
        # regular L1 cycle-consistency
        cycle_loss_L1 = self.criterion(reconstructed, real)

        # cycle-consistency using a weighted combination of SSIM and L1
        if self.ssim_criterion:
            # Data range needs to be positive and normalized
            # https://github.com/VainF/pytorch-msssim#2-normalized-input
            ssim_real = (real + 1) / 2
            ssim_reconstructed = (reconstructed + 1) / 2

            # SSIM criterion returns distance metric
            cycle_loss_ssim = self.ssim_criterion(ssim_reconstructed, ssim_real, data_range=1)

            # weighted sum of SSIM and L1 losses for both forward and backward cycle losses
            return self.alpha * cycle_loss_ssim + self.beta * cycle_loss_L1
        else:
            return cycle_loss_L1


class IdentityLoss:

    def __init__(self, lambda_identity):
        self.lambda_identity = lambda_identity
        self.criterion = torch.nn.L1Loss()

    def __call__(self, idt, real):
        loss_idt = self.criterion(idt, real)
        return loss_idt * self.lambda_identity


if __name__ == '__main__':
    # Initialize CycleLoss with 50% SSIM and 50% L1
    cycle_loss = CycleLoss(proportion_ssim=0.5)

    # Example tensors (real and reconstructed images)
    real_image = torch.rand(1, 3, 256, 256)  # Batch of 1, 3 channels, 256x256
    reconstructed_image = torch.rand(1, 3, 256, 256)

    # Compute the cycle-consistency loss
    loss = cycle_loss(real_image, reconstructed_image)
    print("Cycle Loss:", loss.item())