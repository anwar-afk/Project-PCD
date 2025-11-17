import os
import argparse
import glob
import csv
from pathlib import Path
import numpy as np
from skimage import data

from processing import (
    load_image,
    to_grayscale,
    add_gaussian_noise,
    add_salt_pepper,
    histogram_equalization,
    clahe,
    save_image,
)
from utils import (
    sobel_edges,
    prewitt_edges,
    log_edges,
    canny_edges,
    apply_otsu_to_gradient,
    mse,
    psnr,
)


def list_images(folder):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return paths


def process_image(path, out_dir, apply_he=False, apply_clahe=False, save_images=False):
    name = Path(path).stem
    img = load_image(path)
    gray = to_grayscale(img)

    # reference edges from clean image
    ref_sobel = apply_otsu_to_gradient(sobel_edges(gray))
    ref_prewitt = apply_otsu_to_gradient(prewitt_edges(gray))
    ref_log = apply_otsu_to_gradient(log_edges(gray))
    ref_canny = canny_edges(gray)

    results = []

    noise_types = [
        ('clean', None),
        ('gaussian', [0.01, 0.05, 0.1]),
        ('s&p', [0.01, 0.05, 0.1]),
    ]

    for ntype, levels in noise_types:
        if ntype == 'clean':
            imgs = [(None, gray)]
        else:
            imgs = []
            for lv in levels:
                if ntype == 'gaussian':
                    noisy = add_gaussian_noise(gray, var=lv)
                else:
                    noisy = add_salt_pepper(gray, amount=lv)
                imgs.append((lv, noisy))

        for level, simg in imgs:
            proc = simg.copy()
            if apply_he:
                proc = histogram_equalization(proc)
            if apply_clahe:
                proc = clahe(proc)

            # apply edge detectors
            sob = apply_otsu_to_gradient(sobel_edges(proc))
            pre = apply_otsu_to_gradient(prewitt_edges(proc))
            loge = apply_otsu_to_gradient(log_edges(proc))
            can = canny_edges(proc)

            # save example images for visual inspection
            if save_images:
                img_out_dir = os.path.join(out_dir, 'images', name)
                os.makedirs(img_out_dir, exist_ok=True)
                noisy_tag = 'clean' if level is None else f'{ntype}_{level}'
                # save noisy input
                save_image(os.path.join(img_out_dir, f'{name}_{noisy_tag}_input.png'), simg)
                # save enhanced (after HE/CLAHE)
                save_image(os.path.join(img_out_dir, f'{name}_{noisy_tag}_enhanced.png'), proc)
                # save edge maps
                save_image(os.path.join(img_out_dir, f'{name}_{noisy_tag}_sobel.png'), sob)
                save_image(os.path.join(img_out_dir, f'{name}_{noisy_tag}_prewitt.png'), pre)
                save_image(os.path.join(img_out_dir, f'{name}_{noisy_tag}_log.png'), loge)
                save_image(os.path.join(img_out_dir, f'{name}_{noisy_tag}_canny.png'), can)

            # compute metrics vs reference (clean)
            metrics = {
                'image': name,
                'noise_type': ntype,
                'noise_level': '' if level is None else level,
                'sobel_mse': mse(ref_sobel, sob),
                'sobel_psnr': psnr(ref_sobel, sob),
                'prewitt_mse': mse(ref_prewitt, pre),
                'prewitt_psnr': psnr(ref_prewitt, pre),
                'log_mse': mse(ref_log, loge),
                'log_psnr': psnr(ref_log, loge),
                'canny_mse': mse(ref_canny, can),
                'canny_psnr': psnr(ref_canny, can),
            }

            results.append(metrics)

    # save metrics to CSV
    out_metrics = os.path.join(out_dir, f"{name}_metrics.csv")
    os.makedirs(out_dir, exist_ok=True)
    keys = results[0].keys()
    with open(out_metrics, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    return results


def main(args):
    if args.dataset and os.path.isdir(args.dataset):
        imgs = list_images(args.dataset)
    else:
        # use sample
        sample = data.camera()
        sample_path = os.path.join(args.out, 'sample_camera.png')
        os.makedirs(args.out, exist_ok=True)
        from skimage.io import imsave
        imsave(sample_path, sample)
        imgs = [sample_path]

    all_results = []
    for p in imgs:
        r = process_image(p, args.out, apply_he=args.he, apply_clahe=args.clahe)
        all_results.extend(r)

    # write aggregated CSV
    agg_path = os.path.join(args.out, 'metrics.csv')
    if all_results:
        keys = all_results[0].keys()
        with open(agg_path, 'w', newline='', encoding='utf-8') as f:
            import csv
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)

    print('Done. Results in', args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset', help='folder with images')
    parser.add_argument('--out', type=str, default='results', help='output folder')
    parser.add_argument('--he', action='store_true', help='apply histogram equalization')
    parser.add_argument('--clahe', action='store_true', help='apply CLAHE')
    args = parser.parse_args()
    main(args)
