import argparse
import glob
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def test_a2b(input_path, output_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--onnx_model', type=str, default='../checkpoint/cbct2ct.onnx', help='Path to the ONNX model')
    opt = parser.parse_args()
    print(opt)

    session = ort.InferenceSession(opt.onnx_model, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name

    image_files = sorted(glob.glob(os.path.join(input_path, '*.png')))

    remove_and_create_dir(output_path)

    for i, image_file in enumerate(tqdm(image_files, file=sys.stdout)):
        image = Image.open(image_file).convert('L')  # 转换为灰度图像
        image = image.resize((opt.size, opt.size))  # 调整图像大小
        image = np.array(image)  # 转换为 NumPy 数组

        # 归一化并添加 batch 和 channel 维度
        image = (image / 255.0 - 0.5) / 0.5  # 归一化到 [-1, 1]
        image = np.expand_dims(image, axis=0)  # 添加 channel 维度
        image = np.expand_dims(image, axis=0)  # 添加 batch 维度

        # 运行 ONNX 推理
        fake_B = session.run(None, {input_name: image.astype(np.float32)})[0]

        # 反归一化并保存为图像
        fake_B = (fake_B.squeeze() * 0.5 + 0.5) * 255.0  # 反归一化到 [0, 255]
        fake_B = fake_B.astype(np.uint8)  # 转换为 uint8
        Image.fromarray(fake_B).save(os.path.join(output_path, f"{i:04d}.png"))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(image_files)))

    sys.stdout.write('\n')


if __name__ == '__main__':
    input_path = r'../test_data/brain'
    output_path = r'../test_data/brain_a2b_onnx'
    test_a2b(input_path, output_path)
