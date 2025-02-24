import argparse
import os
import shutil
import sys
import time

import numpy as np
import onnxruntime
from tqdm import tqdm

from installer.CBCT2CT import load_data, post_process, save_array_as_nii

def remove_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def val_onnx_files(args):
    args.image_size = 512
    if args.anatomy == 'brain':
        args.image_size = 256
    # args.onnx_path = os.path.join(args.onnx_path, 'cbct2ct.onnx')
    args.onnx_path = os.path.join(args.onnx_path, f'{args.anatomy}.onnx')
    assert os.path.exists(args.cbct_path), f"CBCT file does not exist at {args.cbct_path}"
    print(f"Onnx path: {args.onnx_path}")

    session = onnxruntime.InferenceSession(args.onnx_path)
    shape = [args.image_size, args.image_size]

    # os.makedirs(args.result_path, exist_ok=True)
    remove_and_create_dir(args.result_path)

    for p in os.listdir(args.cbct_path):
        cbct_path = os.path.join(args.cbct_path, p, 'cbct.nii.gz')

        cbct_padded, img_location, origin_cbct = load_data(cbct_path, shape, args.anatomy)

        length = len(cbct_padded)
        cbct_vecs, location_vecs = [], []
        for index in range(length):
            cbct_vecs.append(cbct_padded[index])
            location_vecs.append(img_location)
        cbct_batch = np.array(cbct_vecs[:]).astype(np.float32)
        locations_batch = np.concatenate(location_vecs[:], axis=0).astype(np.float32)

        start_time = time.time()

        out_results = []
        min_v = -1000
        max_v = 2000

        for cbct, image_locations in tqdm(zip(cbct_batch, locations_batch), total=len(cbct_batch), file=sys.stdout):
            cbct = np.expand_dims(cbct, 0)
            cbct = np.expand_dims(cbct, 0)
            output_name = session.get_outputs()[0].name
            ort_inputs = {session.get_inputs()[0].name: (cbct)}

            result = session.run([output_name], ort_inputs)[0]

            out_cal = post_process(result, image_locations, origin_cbct.GetSize(), min_v, max_v)

            out_results.append(np.expand_dims(out_cal, axis=0))

        out_results = np.concatenate(out_results, axis=0)

        os.makedirs(os.path.join(args.result_path, p), exist_ok=True)
        predict_path = os.path.join(args.result_path, p, "predict.nii.gz")

        save_array_as_nii(out_results, predict_path, origin_cbct)
        total_time = time.time() - start_time
        print("time {}s".format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='CBCT2CT.py',
        usage='%(prog)s [options] --cbct_path <path> --mask_path <path> --result_path <path>',
        description="CBCT generates pseudo CT.")
    parser.add_argument('--onnx_path', type=str, default='./installer/checkpoint', help="Path to onnx")
    parser.add_argument('--anatomy', choices=['brain', 'pelvis', 'thorax'], default='brain', help="The anatomy type")
    parser.add_argument('--cbct_path', type=str, default=r'D:\Data\SynthRAD\Task2\brain', help="Path to cbct file")
    parser.add_argument('--result_path', type=str, default='./result', help="Path to save results")
    args = parser.parse_args()

    print(args)
    val_onnx_files(args)
