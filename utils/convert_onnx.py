import argparse
import os
import shutil

import numpy as np
import onnx
import onnxruntime
import torch
from network.models import Generator
from network.unet import UNetGenerator


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_to_onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--anatomy', choices=['brain', 'pelvis'], default='pelvis', help="The anatomy type")
    parser.add_argument('--model_path', type=str, default='../checkpoint', help="Path to save model checkpoints")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG_A2B = UNetGenerator(opt.input_nc, opt.output_nc)
    netG_A2B.to(device)
    weights_A2B = str(os.path.join(opt.model_path, opt.anatomy, 'netG_A2B.pth'))
    netG_A2B.load_state_dict(torch.load(weights_A2B, weights_only=False, map_location='cpu'))

    netG_A2B.eval()
    x = torch.randn(1, opt.input_nc, opt.size, opt.size).to(device)

    torch_out = netG_A2B(x)

    onnx_file_name = str(os.path.join(opt.model_path, opt.anatomy, "cbct2ct.onnx"))

    torch.onnx.export(netG_A2B, x, onnx_file_name, opset_version=11,
                      input_names=["input"], output_names=["output"],
                      verbose=False)
    print(f"Model saved as ONNX to {onnx_file_name}.")

    try:
        onnx_model = onnx.load(onnx_file_name)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model is valid!")
    except Exception as e:
        print(f"Error checking ONNX model: {e}")
        return

    ort_session = onnxruntime.InferenceSession(onnx_file_name)
    for input in ort_session.get_inputs():
        print(f"Input name: {input.name}, shape: {input.shape}")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

    ort_outs = ort_session.run(None, ort_inputs)
    # print(to_numpy(torch_out) - ort_outs[0])
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    shutil.copy(onnx_file_name, f'../installer/checkpoint/{opt.anatomy}.onnx')
    os.remove(onnx_file_name)


if __name__ == '__main__':
    export_to_onnx()
