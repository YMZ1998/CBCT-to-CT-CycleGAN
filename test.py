import argparse
import os
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import NpyDataset
from network.unet import UNetGenerator
from utils.utils import remove_and_create_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataset_path', type=str, default='datasets', help='root directory of the dataset')
    parser.add_argument('--anatomy', choices=['brain', 'pelvis', 'thorax'], default='thorax', help="The anatomy type")
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--model_path', type=str, default='checkpoint', help="Path to save model checkpoints")

    opt = parser.parse_args()
    print(opt)

    opt.model_path = str(os.path.join(opt.model_path, opt.anatomy))
    opt.dataset_path = str(os.path.join(opt.dataset_path, opt.anatomy))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Networks
    netG_A2B = UNetGenerator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = UNetGenerator(opt.output_nc, opt.input_nc).to(device)

    # Load state dicts
    netG_A2B.load_state_dict(
        torch.load(os.path.join(opt.model_path, 'netG_A2B.pth'), weights_only=False, map_location='cpu'))
    netG_B2A.load_state_dict(
        torch.load(os.path.join(opt.model_path, 'netG_B2A.pth'), weights_only=False, map_location='cpu'))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    input_A = torch.zeros(opt.batch_size, opt.input_nc, opt.size, opt.size, device=device, dtype=torch.float32)
    input_B = torch.zeros(opt.batch_size, opt.output_nc, opt.size, opt.size, device=device, dtype=torch.float32)

    # Dataset loader
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize([0.5], [0.5])]
    dataloader = DataLoader(NpyDataset(opt.dataset_path, transforms_=transforms_, mode='test'),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    remove_and_create_dir('./test_data/result/A')
    remove_and_create_dir('./test_data/result/B')

    data_loader_test = tqdm(dataloader, file=sys.stdout)
    i = 0
    for batch in data_loader_test:
        i = i + 1
        # Set model input
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        # Save image files
        save_image(fake_A, './test_data/result/A/%04d.png' % (i + 1))
        save_image(fake_B, './test_data/result/B/%04d.png' % (i + 1))
