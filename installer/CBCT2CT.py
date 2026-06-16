import argparse
import os
import sys
import time

import SimpleITK as sitk
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

ANATOMY_HU_RANGES = {
    'pelvis': (-800, 1500),
    'thorax': (-800, 1500),
    'brain': (-1000, 2000),
}

DEFAULT_IMAGE_SIZES = {
    'pelvis': 512,
    'thorax': 512,
    'brain': 256,
}


def get_hu_range(anatomy):
    try:
        return ANATOMY_HU_RANGES[anatomy]
    except KeyError as exc:
        raise ValueError(f"Unsupported anatomy: {anatomy}") from exc


def resolve_onnx_path(onnx_path, anatomy):
    if os.path.isdir(onnx_path):
        return os.path.join(onnx_path, f'{anatomy}.onnx')
    return onnx_path


def select_providers(provider):
    available = onnxruntime.get_available_providers()

    if provider == 'cpu':
        return ['CPUExecutionProvider']

    if provider == 'cuda':
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError(f"CUDAExecutionProvider is not available. Available providers: {available}")
        providers = ['CUDAExecutionProvider']
        if 'CPUExecutionProvider' in available:
            providers.append('CPUExecutionProvider')
        return providers

    if 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider']
        if 'CPUExecutionProvider' in available:
            providers.append('CPUExecutionProvider')
        return providers

    return ['CPUExecutionProvider']


def create_session(onnx_path, provider='auto'):
    providers = select_providers(provider)
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print("ONNX providers: ", session.get_providers())
    return session


def get_session_image_size(session, fallback_size):
    input_shape = session.get_inputs()[0].shape
    height, width = input_shape[2], input_shape[3]
    if isinstance(height, int) and isinstance(width, int):
        if height != width:
            raise ValueError(f"Only square ONNX inputs are supported, got {input_shape}")
        return height
    return fallback_size


def post_process(out, location, original_size, min_v=-1000, max_v=2000):
    out = np.squeeze(out)
    location = np.squeeze(location)
    if original_size is not None:
        max_shape = max(original_size[0], original_size[1])
        if max_shape > out.shape[0] or max_shape > out.shape[1]:
            out = cv2.resize(out, [max_shape, max_shape], interpolation=cv2.INTER_LINEAR)

    out = (out + 1) / 2
    out = out * (max_v - min_v) + min_v
    out = np.clip(out, min_v, max_v)

    y_min, x_min = int(location[1]), int(location[0])
    y_max, x_max = y_min + int(location[3]), x_min + int(location[2])

    return out[y_min:y_max, x_min:x_max]


def save_array_as_nii(array, file_path, reference=None):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
    if reference is not None:
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)


def img_normalize(img, anatomy):
    min_v, max_v = get_hu_range(anatomy)
    img = np.clip(img, min_v, max_v)

    min_value = np.min(img)
    max_value = np.max(img)
    print("min_value: ", min_value, "max_value: ", max_value)

    img = (img - min_v) / (max_v - min_v)
    img = img * 2 - 1
    return img


def img_padding(img, x, y, v=0):
    h, w = img.shape[1], img.shape[2]

    padding_y = (y - h) // 2, (y - h) - (y - h) // 2
    padding_x = (x - w) // 2, (x - w) - (x - w) // 2

    padded_img = np.pad(img, ((0, 0), padding_y, padding_x), mode='constant', constant_values=v)

    img_location = np.array([padding_x[0], padding_y[0], w, h])
    img_location = np.expand_dims(img_location, 0)
    return padded_img, img_location


def load_data(cbct_path, shape, anatomy, debug_dir=None):
    origin_cbct = sitk.ReadImage(cbct_path)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        sitk.WriteImage(origin_cbct, os.path.join(debug_dir, "origin_cbct.nii.gz"))

    cbct_array = sitk.GetArrayFromImage(origin_cbct)
    original_size = origin_cbct.GetSize()
    print("Original size: ", original_size)
    original_spacing = origin_cbct.GetSpacing()
    print("Original spacing: ", original_spacing)

    if int(np.min(cbct_array)) == 0:
        cbct_array = cbct_array - 1000

    if cbct_array.shape[1] > shape[0] or cbct_array.shape[2] > shape[1]:
        print("Resampling CBCT...")
        max_shape = max(cbct_array.shape[1], cbct_array.shape[2])
        cbct_array, img_location = img_padding(cbct_array, max_shape, max_shape, np.min(cbct_array))
        padding_cbct = sitk.GetImageFromArray(cbct_array)
        padding_cbct.SetSpacing(original_spacing)
        padding_cbct.SetOrigin(origin_cbct.GetOrigin())
        padding_cbct.SetDirection(origin_cbct.GetDirection())

        # Keep the in-plane physical size consistent after resizing.
        new_spacing = [
            original_spacing[0] * cbct_array.shape[1] / shape[0],
            original_spacing[1] * cbct_array.shape[2] / shape[1],
            original_spacing[2],
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([shape[0], shape[1], cbct_array.shape[0]])
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(padding_cbct.GetOrigin())
        resampler.SetOutputDirection(padding_cbct.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        cbct_resampled = resampler.Execute(padding_cbct)

        cbct_array = sitk.GetArrayFromImage(cbct_resampled)
        print("Resampled cbct shape: ", cbct_array.shape)
        print("New spacing: ", new_spacing)
        cbct_array = img_normalize(cbct_array, anatomy)
    else:
        cbct_array = img_normalize(cbct_array, anatomy)
        cbct_array, img_location = img_padding(cbct_array, shape[0], shape[1], -1)

    return cbct_array, img_location, origin_cbct


def infer_volume(session, cbct_padded, img_location, origin_cbct, anatomy):
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    min_v, max_v = get_hu_range(anatomy)

    out_results = []
    image_location = np.squeeze(img_location).astype(np.float32)
    for cbct in tqdm(cbct_padded.astype(np.float32), total=len(cbct_padded), file=sys.stdout):
        cbct = np.expand_dims(cbct, 0)
        cbct = np.expand_dims(cbct, 0)
        result = session.run([output_name], {input_name: cbct})[0]
        out_cal = post_process(result, image_location, origin_cbct.GetSize(), min_v, max_v)
        out_results.append(np.expand_dims(out_cal, axis=0))

    out_results = np.concatenate(out_results, axis=0)
    expected_shape = (origin_cbct.GetSize()[2], origin_cbct.GetSize()[1], origin_cbct.GetSize()[0])
    if out_results.shape != expected_shape:
        raise ValueError(f"Output shape {out_results.shape} does not match input image shape {expected_shape}")
    return out_results


def val_onnx(args):
    args.onnx_path = resolve_onnx_path(args.onnx_path, args.anatomy)
    assert os.path.exists(args.cbct_path), f"CBCT file does not exist at {args.cbct_path}"
    assert os.path.exists(args.onnx_path), f"Onnx file does not exist at {args.onnx_path}"

    print(f"Onnx path: {args.onnx_path}")
    session = create_session(args.onnx_path, getattr(args, 'provider', 'auto'))
    fallback_size = getattr(args, 'image_size', None) or DEFAULT_IMAGE_SIZES[args.anatomy]
    args.image_size = get_session_image_size(session, fallback_size)
    shape = [args.image_size, args.image_size]

    os.makedirs(args.result_path, exist_ok=True)
    debug_dir = args.result_path if getattr(args, 'debug', False) else None
    cbct_padded, img_location, origin_cbct = load_data(args.cbct_path, shape, args.anatomy, debug_dir=debug_dir)

    start_time = time.time()
    out_results = infer_volume(session, cbct_padded, img_location, origin_cbct, args.anatomy)

    predict_path = os.path.join(args.result_path, args.file_name)
    save_array_as_nii(out_results, predict_path, origin_cbct)
    total_time = time.time() - start_time
    print(f"Saved prediction: {predict_path}")
    print("time {}s".format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='CBCT2CT.py',
        usage='%(prog)s [options] --cbct_path <path> --result_path <path>',
        description="CBCT generates pseudo CT.")
    parser.add_argument('--onnx_path', type=str, default='./checkpoint', help="Path to onnx file or directory")
    parser.add_argument('--anatomy', choices=['brain', 'pelvis', 'thorax'], default='pelvis', help="The anatomy type")
    # parser.add_argument('--cbct_path', type=str, default='./dist/test_data/cbct.nii.gz', help="Path to cbct file")
    parser.add_argument('--cbct_path', type=str, default=r"D:\Data\cbct\denoise_output.mhd", help="Path to cbct file")
    # parser.add_argument('--cbct_path', type=str, default=r"E:\Data\synthRAD2025_Task2_Train\Task2\TH\2THA005\cbct.mha", help="Path to cbct file")
    parser.add_argument('--result_path', type=str, default='./result', help="Path to save results")
    parser.add_argument('--file_name', type=str, default='predict.nii.gz', help="Prediction file name")
    parser.add_argument('--image_size', type=int, default=None, help="Fallback image size if ONNX input is dynamic")
    parser.add_argument('--provider', choices=['auto', 'cpu', 'cuda'], default='auto', help="ONNXRuntime provider")
    parser.add_argument('--debug', action='store_true', default=True, help="Save intermediate debug images")
    args = parser.parse_args()

    print(args)
    val_onnx(args)
