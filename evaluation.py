import argparse
import os
import time

from installer.CBCT2CT import (
    DEFAULT_IMAGE_SIZES,
    create_session,
    get_session_image_size,
    infer_volume,
    load_data,
    resolve_onnx_path,
    save_array_as_nii,
)


def find_cbct_file(case_dir):
    for file_name in ('cbct.nii.gz', 'cbct.mha', 'cbct.nii', 'cbct.mhd'):
        cbct_path = os.path.join(case_dir, file_name)
        if os.path.exists(cbct_path):
            return cbct_path
    return None


def val_onnx_files(args):
    args.onnx_path = resolve_onnx_path(args.onnx_path, args.anatomy)
    assert os.path.exists(args.cbct_path), f"CBCT directory does not exist at {args.cbct_path}"
    assert os.path.exists(args.onnx_path), f"Onnx file does not exist at {args.onnx_path}"
    print(f"Onnx path: {args.onnx_path}")

    session = create_session(args.onnx_path, args.provider)
    fallback_size = args.image_size or DEFAULT_IMAGE_SIZES[args.anatomy]
    args.image_size = get_session_image_size(session, fallback_size)
    shape = [args.image_size, args.image_size]

    os.makedirs(args.result_path, exist_ok=True)

    for case_name in sorted(os.listdir(args.cbct_path)):
        case_dir = os.path.join(args.cbct_path, case_name)
        if not os.path.isdir(case_dir):
            continue

        cbct_path = find_cbct_file(case_dir)
        if cbct_path is None:
            print(f"Skip {case_name}: no cbct image found")
            continue

        case_result_dir = os.path.join(args.result_path, case_name)
        os.makedirs(case_result_dir, exist_ok=True)
        debug_dir = case_result_dir if args.debug else None

        cbct_padded, img_location, origin_cbct = load_data(cbct_path, shape, args.anatomy, debug_dir=debug_dir)

        start_time = time.time()
        out_results = infer_volume(session, cbct_padded, img_location, origin_cbct, args.anatomy)

        predict_path = os.path.join(case_result_dir, "predict.nii.gz")
        save_array_as_nii(out_results, predict_path, origin_cbct)
        total_time = time.time() - start_time
        print(f"{case_name}: saved {predict_path}, time {total_time}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evaluation.py',
        usage='%(prog)s [options] --cbct_path <directory> --result_path <directory>',
        description="Run CBCT-to-CT ONNX inference for a directory of cases.")
    parser.add_argument('--onnx_path', type=str, default='./installer/checkpoint', help="Path to onnx file or directory")
    parser.add_argument('--anatomy', choices=['brain', 'pelvis', 'thorax'], default='thorax', help="The anatomy type")
    parser.add_argument('--cbct_path', type=str, default=r'D:\Data\SynthRAD\Task2\brain', help="Path to case directory")
    parser.add_argument('--result_path', type=str, default='./result', help="Path to save results")
    parser.add_argument('--image_size', type=int, default=None, help="Fallback image size if ONNX input is dynamic")
    parser.add_argument('--provider', choices=['auto', 'cpu', 'cuda'], default='auto', help="ONNXRuntime provider")
    parser.add_argument('--debug', action='store_true', help="Save intermediate debug images")
    args = parser.parse_args()

    print(args)
    val_onnx_files(args)
