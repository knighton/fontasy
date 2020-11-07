from argparse import ArgumentParser
from collections import defaultdict
import json
import numpy as np
import os
import struct


def parse_args():
    x = ArgumentParser()
    x.add_argument('--dataset', type=str, required=True)
    x.add_argument('--val_frac', type=float, default=0.1)
    return x.parse_args()


def load_img_shape(f):
    x = json.load(open(f))
    c = 1
    h = x['img_height']
    w = x['img_width']
    return c, h, w


def load_dataset(f, img_shape):
    x = np.fromfile(f, np.uint8)
    c, h, w = img_shape
    z = 4 + 4 + c * h * w
    shape = (-1,) + img_shape
    imgs = x.reshape(-1, z)[:, 8:].reshape(shape)
    n = len(imgs)
    x32 = x.view(np.int32).reshape(n, -1)
    font_ids = x32[:, 0]
    char_ids = x32[:, 1]
    return imgs, font_ids, char_ids


def save_split(imgs, font_ids, char_ids, sample_ids, f):
    out = open(f, 'wb')
    for i in sample_ids:
        img = imgs[i]
        font_id = font_ids[i]
        char_id = char_ids[i]
        b = struct.pack('ii', font_id, char_id) + img.tobytes()
        out.write(b)
    out.close()


def main(args):
    assert 0 < args.val_frac < 1

    f = os.path.join(args.dataset, 'meta.json')
    img_shape = load_img_shape(f)

    f = os.path.join(args.dataset, 'all.bin')
    imgs, font_ids, char_ids = load_dataset(f, img_shape)

    font_id2sample_ids = defaultdict(list)
    for sample_id, font_id in enumerate(font_ids):
        font_id2sample_ids[font_id].append(sample_id)

    train_sample_ids = []
    val_sample_ids = []
    for font_id, sample_ids in font_id2sample_ids.items():
        num_val = int(len(sample_ids) * args.val_frac)
        np.random.shuffle(sample_ids)
        train_sample_ids += sample_ids[:-num_val]
        val_sample_ids += sample_ids[-num_val:]
    np.random.shuffle(train_sample_ids)
    np.random.shuffle(val_sample_ids)

    f = os.path.join(args.dataset, 'train.bin')
    save_split(imgs, font_ids, char_ids, train_sample_ids, f)

    f = os.path.join(args.dataset, 'val.bin')
    save_split(imgs, font_ids, char_ids, val_sample_ids, f)


if __name__ == '__main__':
    main(parse_args())
