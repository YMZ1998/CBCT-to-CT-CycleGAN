import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare synthRAD Task2 MHA data for CycleGAN training.')
    parser.add_argument('--input_dir', type=Path, default=r"E:\Data\synthRAD2025_Task2_Train\Task2\AB", help='Folder containing case directories.')
    parser.add_argument('--output_root', type=Path, default=Path('datasets_synthrad2025'),
                        help='Output root. Dataset is written under <output_root>/<dataset_name>.')
    parser.add_argument('--dataset_name', type=str, default='abdomen-512', help='Output dataset folder name.')
    parser.add_argument('--size', type=int, default=512, help='Output axial slice size.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Patient-level train split ratio.')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for patient split.')
    parser.add_argument('--slice_step', type=int, default=3, help='Keep every Nth foreground slice per case.')
    parser.add_argument('--min_mask_pixels', type=int, default=128,
                        help='Minimum mask foreground pixels for keeping a slice.')
    parser.add_argument('--clip_min', type=float, default=-1024, help='Minimum stored HU value.')
    parser.add_argument('--clip_max', type=float, default=2000, help='Maximum stored HU value.')
    parser.add_argument('--overwrite', action='store_true', help='Delete the output dataset if it already exists.')
    return parser.parse_args()


def case_dirs(input_dir):
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and path.name.lower() != 'overviews'
    )


def read_mha(path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def maybe_shift_to_hu(volume):
    if int(np.min(volume)) == 0 and np.max(volume) > 1000:
        return volume.astype(np.float32) - 1000
    return volume.astype(np.float32)


def pad_to_square(image, fill_value):
    height, width = image.shape
    target = max(height, width)
    pad_y = target - height
    pad_x = target - width
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    return np.pad(image, ((top, bottom), (left, right)), mode='constant', constant_values=fill_value)


def resize_slice(image, size, fill_value):
    image = pad_to_square(image, fill_value)
    if image.shape == (size, size):
        resized = image
    else:
        resized = resize(
            image,
            (size, size),
            order=1,
            mode='constant',
            cval=fill_value,
            preserve_range=True,
            anti_aliasing=True,
        )
    return resized


def prepare_hu_slice(image, size, clip_min, clip_max):
    fill_value = min(float(np.min(image)), -1000.0)
    image = resize_slice(image, size, fill_value)
    image = np.clip(image, clip_min, clip_max)
    return np.rint(image).astype(np.int16)


def selected_slices(mask, min_mask_pixels, slice_step):
    indices = [idx for idx in range(mask.shape[0]) if np.count_nonzero(mask[idx]) >= min_mask_pixels]
    return indices[::slice_step]


def split_cases(cases, train_ratio, seed):
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * train_ratio)
    return sorted(shuffled[:split_at]), sorted(shuffled[split_at:])


def make_output_dirs(output_dir, overwrite):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f'{output_dir} already exists. Re-run with --overwrite to replace it.')
        shutil.rmtree(output_dir)

    for split in ('train', 'test'):
        for domain in ('A', 'B'):
            (output_dir / split / domain).mkdir(parents=True, exist_ok=True)


def process_case(case_dir, split, output_dir, args):
    cbct_path = case_dir / 'cbct.mha'
    ct_path = case_dir / 'ct.mha'
    mask_path = case_dir / 'mask.mha'

    if not cbct_path.exists() or not ct_path.exists() or not mask_path.exists():
        return {'case': case_dir.name, 'saved': 0, 'skipped': 'missing cbct/ct/mask'}

    cbct = maybe_shift_to_hu(read_mha(cbct_path))
    ct = maybe_shift_to_hu(read_mha(ct_path))
    mask = read_mha(mask_path)

    if cbct.shape != ct.shape or cbct.shape != mask.shape:
        return {'case': case_dir.name, 'saved': 0, 'skipped': f'shape mismatch {cbct.shape}/{ct.shape}/{mask.shape}'}

    indices = selected_slices(mask, args.min_mask_pixels, args.slice_step)
    saved = 0
    for slice_idx in indices:
        cbct_slice = prepare_hu_slice(cbct[slice_idx], args.size, args.clip_min, args.clip_max)
        ct_slice = prepare_hu_slice(ct[slice_idx], args.size, args.clip_min, args.clip_max)
        file_name = f'{case_dir.name}_{slice_idx:03d}.npy'
        np.save(output_dir / split / 'A' / file_name, cbct_slice)
        np.save(output_dir / split / 'B' / file_name, ct_slice)
        saved += 1

    return {'case': case_dir.name, 'saved': saved, 'skipped': None}


def count_files(output_dir):
    counts = {}
    for split in ('train', 'test'):
        for domain in ('A', 'B'):
            counts[f'{split}/{domain}'] = len(list((output_dir / split / domain).glob('*.npy')))
    return counts


def main():
    args = parse_args()
    output_dir = args.output_root / args.dataset_name

    cases = case_dirs(args.input_dir)
    if not cases:
        raise FileNotFoundError(f'No case directories found in {args.input_dir}')

    train_cases, test_cases = split_cases(cases, args.train_ratio, args.seed)
    make_output_dirs(output_dir, args.overwrite)

    metadata = {
        'input_dir': str(args.input_dir),
        'output_dir': str(output_dir),
        'size': args.size,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'slice_step': args.slice_step,
        'min_mask_pixels': args.min_mask_pixels,
        'clip_min': args.clip_min,
        'clip_max': args.clip_max,
        'num_cases': len(cases),
        'train_cases': [case.name for case in train_cases],
        'test_cases': [case.name for case in test_cases],
        'case_results': [],
    }

    for split, split_cases_ in (('train', train_cases), ('test', test_cases)):
        for case_dir in tqdm(split_cases_, desc=f'Processing {split}', file=sys.stdout):
            metadata['case_results'].append(process_case(case_dir, split, output_dir, args))

    metadata['counts'] = count_files(output_dir)
    metadata_path = output_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(json.dumps(metadata['counts'], indent=2))
    print(f'Dataset saved to {output_dir}')
    print(f'Metadata saved to {metadata_path}')


if __name__ == '__main__':
    main()
