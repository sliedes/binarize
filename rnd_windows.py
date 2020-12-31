#!/usr/bin/env python3

import sys
import random
import os
import warnings
import argparse

import numpy as np

import skimage.io
import skimage.transform
import skimage.util

import cv2

SIZE = 224
AREA_COVERAGE = 1  # 5

def pad_im(im):
    pad_width = [(0, 0)] * len(im.shape)
    pad_width[0] = pad_width[1] = (SIZE // 2, SIZE // 2)
    return np.pad(im, pad_width, mode='reflect')  # constant_values=255.0)

def rnd_window(im):
    # print('im shape:', im.shape)
    pad = SIZE // 2
    yoff = random.randint(pad, im.shape[0] - SIZE - 1)
    xoff = random.randint(pad, im.shape[1] - SIZE - 1)
    # print('window corner:', xoff, yoff)
    win = im[yoff - pad:yoff + SIZE + pad, xoff - pad:xoff + SIZE + pad]
    # print(win.shape)
    winr = skimage.transform.rotate(win, random.random() * 360, resize=True, mode='reflect')
    # print(winr.shape)
    winr = winr[(winr.shape[0] - SIZE) // 2:, (winr.shape[1] - SIZE) // 2:]
    # print(winr.shape)
    winr = winr[:-(winr.shape[0] - SIZE), :-(winr.shape[1] - SIZE)]
    # print(winr.shape)
    assert winr.shape[0] == SIZE and winr.shape[1] == SIZE, winr.shape
    return winr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--allow-no-mask', action='store_true', help="Don't fail if no mask file provided")
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('src', type=str, help='Source image')
    args = parser.parse_args()

    if args.output_dir is None:
        print("Error: Need --output-directory", file=sys.stderr)
        sys.exit(1)

    mask_fname = os.path.join(os.path.dirname(args.src), 'fg_mask', os.path.basename(args.src))

    im = skimage.util.img_as_float(skimage.io.imread(args.src))

    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]

    try:
        mask = skimage.util.img_as_float(skimage.io.imread(mask_fname))
    except FileNotFoundError:
        if not args.allow_no_mask:
            raise
        mask = None

    if mask is not None:
        assert im.shape[:2] == mask.shape, (im.shape, mask.shape)
        assert len(mask.shape) == 2
        mask = mask[:, :, np.newaxis]
        im = np.concatenate([im, mask], axis=2)

    if args.scale != 1.0:
        im = cv2.resize(im, dsize=(int(round(im.shape[1]*args.scale)), int(round(im.shape[0]*args.scale))),
                        interpolation=cv2.INTER_CUBIC)
        im = np.clip(im, 0.0, 1.0)

    if im.shape[0] < SIZE or im.shape[1] < SIZE:
        print('Image too small to extract windows. Bailing out.')
        sys.exit(0)

    num_windows = int((im.shape[0] * im.shape[1] / (SIZE * SIZE)) * AREA_COVERAGE + .5)
    print(f'Extracting {num_windows} windows.')

    orig_dir = os.path.join(args.output_dir, 'orig')
    mask_dir = os.path.join(args.output_dir, 'mask')

    os.makedirs(orig_dir, exist_ok=True)

    if mask is not None:
        os.makedirs(mask_dir, exist_ok=True)

    base = os.path.basename(args.src).rsplit('.', 1)[0]
    if args.scale != 1.0:
        base += f'_scale{args.scale}'

    im = pad_im(im)

    for i in range(num_windows):
        print('.', end='', flush=True)
        win = rnd_window(im)
        if random.random() < .5:
            win = win[::-1]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.* is a low contrast image.*', category=UserWarning)
            if mask is not None:
                skimage.io.imsave(os.path.join(mask_dir, f'{base}_{i}.png'), skimage.util.img_as_ubyte(win[:, :, -1]))
                win = win[:, :, :-1]
            skimage.io.imsave(os.path.join(orig_dir, f'{base}_{i}.png'), skimage.util.img_as_ubyte(win))

    print()

if __name__ == '__main__':
    main()
