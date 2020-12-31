#!/usr/bin/env python3

'''Apply segmentation to all .pngs in a directory'''

import sys
import os
import numpy as np

import tensorflow as tf

from sklearn.isotonic import IsotonicRegression

SIZE = 224
STRIDE = 224

def load_image(fname):
    return tf.cast(tf.image.decode_png(
        tf.io.read_file(fname), channels=3), tf.float32).numpy()

def iter_strides(shape):
    for y in range(0, shape[0] - SIZE, STRIDE):
        for x in range(0, shape[1] - SIZE, STRIDE):
            yield y, x

def img_to_slices_one_dir(im):
    # cuts from right/bottom, does not necessarily include entire image
    assert im.shape[0] >= SIZE, im.shape
    assert im.shape[1] >= SIZE, im.shape

    slices = []
    for y, x in iter_strides(im.shape):
        slices.append(im[y:y+SIZE, x:x+SIZE])
    return np.array(slices)

def slices_to_img_one_dir(slices, shape):
    count = np.zeros(shape)
    data = np.zeros(shape)

    h = shape[0] - shape[0] % SIZE
    w = shape[1] - shape[1] % SIZE

    for slc, (y, x) in zip(slices, iter_strides(shape)):
        data[y:y+SIZE, x:x+SIZE] += slc
        count[y:y+SIZE, x:x+SIZE] += 1

    return data, count

def img_to_slices(im):
    slices = [
        img_to_slices_one_dir(im),
        img_to_slices_one_dir(im[::-1]),
        img_to_slices_one_dir(im[:, ::-1]),
        img_to_slices_one_dir(im[::-1, ::-1])]
    return np.concatenate(slices, axis=0)

def slices_to_img(slices, shape):
    shape = shape[:2] + slices.shape[3:]
    count = np.zeros(shape)
    data = np.zeros(shape)

    assert len(slices) % 4 == 0, slices.shape
    slices_per_dir = len(slices) // 4

    d, c = slices_to_img_one_dir(slices[0:slices_per_dir], shape)
    data += d
    count += c

    d, c = slices_to_img_one_dir(slices[slices_per_dir:slices_per_dir*2], shape)
    data += d[::-1]
    count += c[::-1]

    d, c = slices_to_img_one_dir(slices[slices_per_dir*2:slices_per_dir*3], shape)
    data += d[:, ::-1]
    count += c[:, ::-1]

    d, c = slices_to_img_one_dir(slices[slices_per_dir*3:], shape)
    data += d[::-1, ::-1]
    count += c[::-1, ::-1]

    assert np.min(count) > 0, np.min(count)
    return data / count

def main():
    src_dir = sys.argv[1]

    if src_dir.endswith('.png'):
        assert os.path.isfile(src_dir)
        src_fnames = [src_dir]
        src_dir = os.path.dirname(src_dir)
        tgt_dir = os.path.join(src_dir, 'auto_bin')
    else:
        assert os.path.isdir(src_dir)
        # ignore empty auto_seg dir
        tgt_dir = os.path.join(src_dir, 'auto_bin')
        try:
            os.rmdir(tgt_dir)
        except FileNotFoundError:
            pass
        os.makedirs(tgt_dir)
        src_fnames = [os.path.join(src_dir, x) for x in sorted(os.listdir(src_dir)) if x.lower().endswith('.png')]

    tgt_fnames = [os.path.join(tgt_dir, os.path.basename(x)) for x in src_fnames]

    if not src_fnames:
        print('No .pngs in given directory.', file=sys.stderr)
        sys.exit(1)

    # src_fnames_ds = tf.data.Dataset.from_tensor_slices(src_fnames)
    # src_imgs = src_fnames_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.load_model('model')

    isoregr_params = np.load('model/isoregr.npz')
    isoregr = IsotonicRegression().fit(isoregr_params['X'], isoregr_params['y'])

    for src_fname, tgt_fname in zip(src_fnames, tgt_fnames):
        im = load_image(src_fname)
        print(src_fname, im.shape)
        orig_shape = im.shape
        if im.shape[0] < SIZE or im.shape[1] < SIZE:
            print(f'ERROR: {src_fname}: at least one dimension smaller than {SIZE}, skipping image.', file=sys.stderr)
            continue
        im = img_to_slices(im)
        for i in range(0, len(im), 64):
            chunk = model.predict(im[i:i+64])
            sh = chunk.shape
            chunk = tf.math.sigmoid(chunk.flatten())
            chunk = isoregr.predict(chunk).reshape(sh)
            im[i:i+64] = chunk
        im = slices_to_img(im, orig_shape)
        im = (np.clip(im, 0.0, 1.0) * 255.0).astype(np.uint8)
        tf.io.write_file(tgt_fname, tf.io.encode_png(im))

if __name__ == '__main__':
    main()
