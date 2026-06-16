import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network.unet import ResUNetGenerator, UNetGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Export CBCT-to-CT generator checkpoint to ONNX.')
    parser.add_argument('--anatomy', choices=['brain', 'pelvis', 'thorax'], default='pelvis', help='anatomy type')
    parser.add_argument('--size', type=int, default=512, help='fixed square input size')
    parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
    parser.add_argument('--generator', choices=['resunet', 'unet'], default='resunet', help='generator architecture')
    parser.add_argument('--ngf', type=int, default=16, help='base number of generator filters')
    parser.add_argument('--model_path', type=Path, default=Path('checkpoint'), help='checkpoint root')
    parser.add_argument('--experiment_name', type=str, default=None, help='checkpoint subfolder name')
    parser.add_argument('--weights', type=Path, default=None, help='direct path to netG_A2B.pth')
    parser.add_argument('--output_path', type=Path, default=None, help='output ONNX path')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='export device')
    parser.add_argument('--seed', type=int, default=2025, help='random seed for verification input')
    parser.add_argument('--rtol', type=float, default=1e-3, help='verification relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-4, help='verification absolute tolerance')
    parser.add_argument('--copy_to_dist', action='store_true', help='also copy ONNX to installer/dist/checkpoint')
    return parser.parse_args()


def default_experiment_name(args):
    return f'datasets_synthrad2025-{args.anatomy}-{args.size}-{args.generator}-ngf{args.ngf}'


def resolve_weights_path(args):
    if args.weights:
        return resolve_path(args.weights)

    experiment_name = args.experiment_name or default_experiment_name(args)
    return resolve_path(args.model_path) / experiment_name / 'netG_A2B.pth'


def resolve_output_path(args):
    if args.output_path:
        return resolve_path(args.output_path)
    return PROJECT_ROOT / 'installer' / 'checkpoint' / f'{args.anatomy}.onnx'


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_generator(args):
    if args.generator == 'resunet':
        return ResUNetGenerator(args.input_nc, args.output_nc, ngf=args.ngf)
    return UNetGenerator(args.input_nc, args.output_nc, ngf=args.ngf)


def load_state_dict(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def export_to_onnx():
    args = parse_args()
    weights_path = resolve_weights_path(args)
    output_path = resolve_output_path(args)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if not weights_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {weights_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Weights: {weights_path}')
    print(f'Output:  {output_path}')
    print(f'Model:   {args.generator}, ngf={args.ngf}, size={args.size}')
    print(f'Device:  {device}')

    torch.manual_seed(args.seed)
    netG_A2B = build_generator(args)
    netG_A2B.load_state_dict(load_state_dict(weights_path))
    netG_A2B.to(device)
    netG_A2B.eval()

    x = torch.randn(1, args.input_nc, args.size, args.size, device=device, dtype=torch.float32)

    with torch.inference_mode():
        torch_out = netG_A2B(x)

    torch.onnx.export(
        netG_A2B,
        x,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        verbose=False,
    )
    print(f'ONNX saved: {output_path}')

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print('ONNX checker: OK')

    ort_session = onnxruntime.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    for item in ort_session.get_inputs():
        print(f'Input:  {item.name}, shape={item.shape}, type={item.type}')
    for item in ort_session.get_outputs():
        print(f'Output: {item.name}, shape={item.shape}, type={item.type}')

    ort_out = ort_session.run(None, {ort_session.get_inputs()[0].name: to_numpy(x)})[0]
    torch_out_np = to_numpy(torch_out)
    diff = np.abs(torch_out_np - ort_out)
    print(f'max abs diff:  {diff.max():.8f}')
    print(f'mean abs diff: {diff.mean():.8f}')
    np.testing.assert_allclose(torch_out_np, ort_out, rtol=args.rtol, atol=args.atol)
    print('PyTorch vs ONNXRuntime: OK')

    if args.copy_to_dist:
        dist_output = PROJECT_ROOT / 'installer' / 'dist' / 'checkpoint' / f'{args.anatomy}.onnx'
        dist_output.parent.mkdir(parents=True, exist_ok=True)
        dist_output.write_bytes(output_path.read_bytes())
        print(f'Copied to: {dist_output}')


if __name__ == '__main__':
    export_to_onnx()
