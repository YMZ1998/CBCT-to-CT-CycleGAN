import argparse
import os
import shutil
import sys
import time

import SimpleITK as sitk
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm


def post_process(out, location, original_size, min_v=-1000, max_v=2000):
    out = np.squeeze(out)
    location = np.squeeze(location)
    if original_size is not None:
        max_shape = max(original_size[0], original_size[1])
        if max_shape > out.shape[0] or max_shape > out.shape[1]:
            out = cv2.resize(out, [max_shape, max_shape], interpolation=cv2.INTER_LINEAR)
            # sitk.WriteImage(sitk.GetImageFromArray(out), os.path.join(args.result_path, "out_resampled.nii.gz"))

    out = (out + 1) / 2
    out = out * (max_v - min_v) + min_v
    out = np.clip(out, min_v, max_v)

    y_min, x_min = int(location[1]), int(location[0])
    y_max, x_max = y_min + int(location[3]), x_min + int(location[2])

    out = out[y_min:y_max, x_min:x_max]

    return out


def save_array_as_nii(array, file_path, reference=None):
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)
    if reference is not None:
        sitk_image.CopyInformation(reference)
    sitk.WriteImage(sitk_image, file_path)


def img_normalize(img, anatomy):
    if anatomy == 'pelvis':
        min_v = -800
        max_v = 1500
    elif anatomy == 'chest':
        min_v = -800
        max_v = 1500
    elif anatomy == 'brain':
        min_v = -1000
        max_v = 2000
    img = np.clip(img, min_v, max_v)

    min_value = np.min(img)
    max_value = np.max(img)
    print("min_value: ", min_value, "max_value: ", max_value)
    # img = (img - min_value) / (max_value - min_value)
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


def load_data(cbct_path, shape, anatomy):
    origin_cbct = sitk.ReadImage(cbct_path)
    cbct_array = sitk.GetArrayFromImage(origin_cbct)
    original_size = origin_cbct.GetSize()
    print("Original size: ", original_size)
    original_spacing = origin_cbct.GetSpacing()
    print("Original spacing: ", original_spacing)

    if int(np.min(cbct_array)) == 0:
        cbct_array = cbct_array - 1000

    # 如果CBCT尺寸大于目标尺寸，进行重采样
    if cbct_array.shape[1] > shape[0] or cbct_array.shape[2] > shape[1]:
        print("Resampling CBCT...")
        max_shape = max(cbct_array.shape[1], cbct_array.shape[2])
        # print("Max shape: ", max_shape)
        # print(np.min(cbct_array))
        cbct_array, img_location = img_padding(cbct_array, max_shape, max_shape, np.min(cbct_array))
        padding_cbct = sitk.GetImageFromArray(cbct_array)
        padding_cbct.SetSpacing(original_spacing)
        # sitk.WriteImage(padding_cbct, os.path.join(args.result_path, "padding_cbct.nii.gz"))

        # 计算新的spacing保持物理尺寸一致
        new_spacing = [
            original_spacing[0] * cbct_array.shape[1] / shape[0],
            original_spacing[1] * cbct_array.shape[2] / shape[1],
            original_spacing[2]  # Z轴不变
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([shape[0], shape[1], cbct_array.shape[0]])
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        cbct_resampled = resampler.Execute(padding_cbct)

        cbct_array = sitk.GetArrayFromImage(cbct_resampled)
        print("Resampled cbct shape: ", cbct_array.shape)
        print("New spacing: ", new_spacing)
        # sitk.WriteImage(cbct_resampled, os.path.join(args.result_path, "resample_cbct.nii.gz"))

        cbct_array = img_normalize(cbct_array, anatomy)

    else:
        cbct_array = img_normalize(cbct_array, anatomy)
        cbct_array, img_location = img_padding(cbct_array, shape[0], shape[1], -1)

    return cbct_array, img_location, origin_cbct


def val_onnx(args):
    args.image_size = 256
    # args.onnx_path = os.path.join(args.onnx_path, 'cbct2ct.onnx')
    args.onnx_path = os.path.join(args.onnx_path, f'{args.anatomy}.onnx')
    assert os.path.exists(args.cbct_path), f"CBCT file does not exist at {args.cbct_path}"
    print(f"Onnx path: {args.onnx_path}")
    shape = [args.image_size, args.image_size]

    os.makedirs(args.result_path, exist_ok=True)
    assert os.path.exists(args.cbct_path), f"CBCT file does not exist at {args.cbct_path}"
    assert os.path.exists(args.onnx_path), f"Onnx file does not exist at {args.onnx_path}"

    cbct_padded, img_location, origin_cbct = load_data(args.cbct_path, shape, args.anatomy)

    length = len(cbct_padded)
    cbct_vecs, location_vecs = [], []
    for index in range(length):
        cbct_vecs.append(cbct_padded[index])
        location_vecs.append(img_location)
    cbct_batch = np.array(cbct_vecs[:]).astype(np.float32)
    locations_batch = np.concatenate(location_vecs[:], axis=0).astype(np.float32)

    session = onnxruntime.InferenceSession(args.onnx_path)

    # providers = [
    #     'CUDAExecutionProvider',
    #     'TensorrtExecutionProvider',
    #     'CPUExecutionProvider'
    # ]
    #
    # session = onnxruntime.InferenceSession(args.onnx_path, providers=providers)
    # print(onnxruntime.get_available_providers())

    start_time = time.time()

    out_results = []
    min_v = -1000
    max_v = 2000
    # if args.anatomy == 'pelvis':
    #     min_v = -1000
    #     max_v = 2000
    # elif args.anatomy == 'brain':
    #     min_v = -1000
    #     max_v = 2000
    for cbct, image_locations in tqdm(zip(cbct_batch, locations_batch), total=len(cbct_batch), file=sys.stdout):
        # cbct = img_normalize(cbct, args.anatomy)
        cbct = np.expand_dims(cbct, 0)
        cbct = np.expand_dims(cbct, 0)
        output_name = session.get_outputs()[0].name
        ort_inputs = {session.get_inputs()[0].name: (cbct)}

        result = session.run([output_name], ort_inputs)[0]

        out_cal = post_process(result, image_locations, origin_cbct.GetSize(), min_v, max_v)

        out_results.append(np.expand_dims(out_cal, axis=0))

    out_results = np.concatenate(out_results, axis=0)

    predict_path = os.path.join(args.result_path, "predict.nii.gz")

    save_array_as_nii(out_results, predict_path, origin_cbct)
    total_time = time.time() - start_time
    print("time {}s".format(total_time))


def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    # CBCT2CT.exe  --cbct_path ./test_data/cbct.nii.gz --mask_path ./test_data/mask.nii.gz --result_path ./result --anatomy brain
    # CBCT2CT.exe --cbct_path ./test_data/brain/cbct.nii.gz --mask_path ./test_data/brain/mask.nii.gz --result_path ./result --anatomy brain
    # CBCT2CT.exe --cbct_path ./test_data/pelvis/cbct.nii.gz --mask_path ./test_data/pelvis/mask.nii.gz --result_path ./result --anatomy pelvis
    parser = argparse.ArgumentParser(
        prog='CBCT2CT.py',
        usage='%(prog)s [options] --cbct_path <path> --mask_path <path> --result_path <path>',
        description="CBCT generates pseudo CT.")
    parser.add_argument('--onnx_path', type=str, default='./checkpoint', help="Path to onnx")
    parser.add_argument('--anatomy', choices=['brain', 'pelvis', 'chest'], default='chest', help="The anatomy type")
    parser.add_argument('--cbct_path', type=str, default='../test_data/pelvis5.nii.gz', help="Path to cbct file")
    # parser.add_argument('--cbct_path', type=str, default='../test_data/brain_1/cbct.nii.gz', help="Path to cbct file")
    # parser.add_argument('--mask_path', type=str, required=True, help="Path to mask file")
    parser.add_argument('--result_path', type=str, default='../test_data', help="Path to save results")
    # parser.add_argument('--debug', type=bool, default=False, help="Debug options")
    args = parser.parse_args()

    # args.onnx_path = str(os.path.join(args.onnx_path, args.anatomy))
    print(args)
    val_onnx(args)
