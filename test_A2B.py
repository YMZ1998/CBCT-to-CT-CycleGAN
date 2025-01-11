import argparse
import glob
import os
import shutil
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from network.models import Generator


class ImageDataset(Dataset):
    def __init__(self, folder, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(folder, '*.*')))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        return {'A': item_A}

    def __len__(self):
        return len(self.files_A)


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def test_a2b(input_path, output_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--anatomy', choices=['brain', 'pelvis'], default='pelvis', help="The anatomy type")
    parser.add_argument('--model_path', type=str, default='checkpoint', help="Path to save model checkpoints")

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    netG_A2B = Generator(opt.input_nc, opt.output_nc)

    if opt.cuda:
        netG_A2B.cuda()

    weights_A2B = str(os.path.join(opt.model_path, opt.anatomy, 'netG_A2B.pth'))
    netG_A2B.load_state_dict(torch.load(weights_A2B, weights_only=False, map_location='cpu'))

    netG_A2B.eval()

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize([0.5], [0.5])]
    dataloader = DataLoader(ImageDataset(input_path, transforms_=transforms_),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    remove_and_create_dir(output_path)
    # remove_and_create_dir('output/B')

    data_loader_test = tqdm(dataloader, file=sys.stdout)
    i = 0
    for batch in data_loader_test:
        i = i + 1
        real_A = Variable(input_A.copy_(batch['A']))

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)

        save_image(fake_B, os.path.join(output_path, f"{i:04d}.png"))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

    sys.stdout.write('\n')


if __name__ == '__main__':
    input_path = r'./test_data/pelvis'
    output_path = r'./test_data/result'
    # output_path = r'output/B'
    test_a2b(input_path, output_path)
