import os
import random
import shutil
import time
import datetime
import sys

from torch import nn
from torch.autograd import Variable
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from visdom import Visdom
except ImportError:
    Visdom = None


def tensor2image(tensor):
    image = tensor[0].detach().cpu().float()
    image = torch.clamp(127.5 * (image + 1.0), 0, 255).numpy()
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


def draw_label(image, label):
    image_hwc = np.transpose(image, (1, 2, 0))
    pil_image = Image.fromarray(image_hwc)
    draw = ImageDraw.Draw(pil_image)
    font_size = max(18, min(image_hwc.shape[:2]) // 22)
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()

    padding = max(6, font_size // 3)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.rectangle(
        (0, 0, text_width + padding * 2, text_height + padding * 2),
        fill=(0, 0, 0),
    )
    draw.text((padding, padding), label, fill=(255, 255, 255), font=font)
    return np.transpose(np.asarray(pil_image), (2, 0, 1))


class Logger():
    def __init__(self, n_epochs, batches_epoch, env='cbct2ct-cyclegan',
                 image_size=256, image_interval=10, plot_interval=10, start_epoch=1):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.start_epoch = start_epoch
        self.env = env
        self.image_size = image_size
        self.image_interval = max(1, image_interval)
        self.plot_interval = max(1, plot_interval)
        self.epoch = start_epoch
        self.batch = 1
        self.global_step = (start_epoch - 1) * batches_epoch
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_names = []
        self.loss_window = 'losses'
        self.a2b_image_window = 'samples_a2b'
        self.b2a_image_window = 'samples_b2a'
        self.status_window = 'status'
        self.image_labels = {
            'real_A': 'real_A | CBCT input',
            'fake_B': 'fake_B | pseudo CT',
            'real_B': 'real_B | CT target',
            'fake_A': 'fake_A | pseudo CBCT',
        }
        self.loss_window_initialized = False
        self.image_windows_initialized = set()
        self.enabled = False

        if Visdom is None:
            print('Visdom is not installed. Training will continue without Visdom logging.')
            return

        self.viz = Visdom(env=self.env)
        if self._check_connection():
            self.enabled = True
            print(f'Visdom logging enabled: http://localhost:8097/env/{self.env}')
        else:
            print('Visdom server is not running. Training will continue without Visdom logging.')

    def _check_connection(self):
        try:
            return self.viz.check_connection(timeout_seconds=3)
        except TypeError:
            return self.viz.check_connection()
        except Exception as exc:
            print(f'Visdom connection check failed: {exc}')
            return False

    @staticmethod
    def _to_scalar(value):
        if hasattr(value, 'detach'):
            return value.detach().cpu().item()
        return float(value)

    def _advance_batch(self):
        if (self.batch % self.batches_epoch) == 0:
            self.epoch += 1
            self.batch = 1
        else:
            self.batch += 1

    def _ordered_images(self, images):
        preferred_order = ['real_A', 'fake_B', 'real_B', 'fake_A']
        ordered = [(name, images[name]) for name in preferred_order if name in images]
        ordered += [(name, tensor) for name, tensor in images.items()
                    if name not in preferred_order]
        return ordered

    def _plot_losses(self, averaged_losses):
        y = np.array([[averaged_losses[name] for name in self.loss_names]])
        x = np.array([[self.global_step] * len(self.loss_names)])
        opts = {
            'title': 'Training Losses',
            'xlabel': 'Iteration',
            'ylabel': 'Loss',
            'legend': self.loss_names,
            'showlegend': True,
        }

        if self.loss_window_initialized:
            self.viz.line(X=x, Y=y, win=self.loss_window, update='append')
        else:
            self.viz.line(X=x, Y=y, win=self.loss_window, opts=opts)
            self.loss_window_initialized = True

    def _show_image_pair(self, window, title, names, images):
        pairs = [(name, images[name]) for name in names if name in images]
        if not pairs:
            return

        image_batch = np.stack([
            draw_label(tensor2image(tensor), self.image_labels.get(name, name))
            for name, tensor in pairs
        ], axis=0)
        opts = None
        if window not in self.image_windows_initialized:
            opts = {
                'title': title,
                'caption': ' | '.join(self.image_labels.get(name, name) for name, _ in pairs),
            }
            self.image_windows_initialized.add(window)

        if opts:
            self.viz.images(image_batch, nrow=len(pairs), win=window, opts=opts)
        else:
            self.viz.images(image_batch, nrow=len(pairs), win=window)

    def _show_images(self, images):
        self._show_image_pair(
            self.a2b_image_window,
            'A2B: CBCT real_A -> pseudo CT fake_B',
            ['real_A', 'fake_B'],
            images,
        )
        self._show_image_pair(
            self.b2a_image_window,
            'B2A: CT real_B -> pseudo CBCT fake_A',
            ['real_B', 'fake_A'],
            images,
        )

    def _show_status(self, averaged_losses, eta):
        rows = ''.join(
            f'<tr><td>{name}</td><td>{value:.6f}</td></tr>'
            for name, value in averaged_losses.items()
        )
        html = f"""
        <h3>CBCT-to-CT CycleGAN</h3>
        <table>
          <tr><td>Environment</td><td>{self.env}</td></tr>
          <tr><td>Epoch</td><td>{self.epoch}/{self.n_epochs}</td></tr>
          <tr><td>Batch</td><td>{self.batch}/{self.batches_epoch}</td></tr>
          <tr><td>Iteration</td><td>{self.global_step}</td></tr>
          <tr><td>ETA</td><td>{eta}</td></tr>
        </table>
        <h4>Epoch averages</h4>
        <table>{rows}</table>
        """
        self.viz.text(html, win=self.status_window, opts={'title': 'Training Status'})

    def log(self, losses=None, images=None):
        losses = losses or {}
        images = images or {}
        self.global_step += 1
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        for loss_name, loss_value in losses.items():
            if loss_name not in self.loss_names:
                self.loss_names.append(loss_name)
            loss_value = self._to_scalar(loss_value)
            if loss_name not in self.losses:
                self.losses[loss_name] = loss_value
            else:
                self.losses[loss_name] += loss_value

        batches_done = self.batches_epoch * (self.epoch - self.start_epoch) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        eta = datetime.timedelta(seconds=batches_left * self.mean_period / max(1, batches_done))
        averaged_losses = {
            loss_name: loss / self.batch
            for loss_name, loss in self.losses.items()
        }

        if self.enabled and averaged_losses:
            try:
                should_plot = (self.batch % self.plot_interval) == 0 or (self.batch % self.batches_epoch) == 0
                if should_plot:
                    self._plot_losses(averaged_losses)
                    self._show_status(averaged_losses, eta)

                if images and (self.batch % self.image_interval) == 0:
                    self._show_images(images)
            except Exception as exc:
                self.enabled = False
                print(f'Visdom logging failed and has been disabled: {exc}')

        if (self.batch % self.batches_epoch) == 0:
            sys.stdout.write(
                '\rEpoch %03d/%03d [%04d/%04d] -- ' %
                (self.epoch, self.n_epochs, self.batch, self.batches_epoch)
            )
            for i, (loss_name, loss) in enumerate(averaged_losses.items()):
                if (i + 1) == len(averaged_losses):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, loss))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, loss))
            sys.stdout.write('ETA: %s' % eta)

            for loss_name in self.losses.keys():
                self.losses[loss_name] = 0.0
            sys.stdout.write('\n')

        self._advance_batch()


class ReplayBuffer():
    def __init__(self, max_size=1000):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        factor = 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )
        return max(0.0, factor)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
