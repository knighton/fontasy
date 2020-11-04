from argparse import ArgumentParser
from collections import Counter
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--out_font_freqs', type=str, required=True)
    x.add_argument('--out_char_freqs', type=str, required=True)
    x.add_argument('--out_heat_txt', type=str, required=True)
    x.add_argument('--out_heat_img', type=str, required=True)
    return x.parse_args()


def plot_id_freqs(ids, f):
    x = Counter(ids)
    x = sorted(x.values())
    plt.plot(x)
    plt.savefig(f)
    plt.clf()


def heatmap_images(images, f_txt, f_img):
    n, c, h, w = images.shape
    x = images.astype(np.uint64).sum(0).sum(0)

    out = open(f_txt, 'w')
    for hh in range(h):
        ss = []
        for ww in range(w):
            count = x[hh, ww] // 1000
            s = '%6d' % count
            ss.append(s)
        line = ' '.join(ss) + '\n'
        out.write(line)
    out.close()

    x = 255 * x / x.max()
    x = x.astype(np.uint8)
    im = Image.fromarray(x)
    im.save(f_img)


def main(args):
    d = getattr(args, 'in')

    f = os.path.join(d, 'meta.json')
    x = json.load(open(f))
    n = x['img_count']
    h = x['img_height']
    w = x['img_width']
    num_fonts = len(x['fonts'])
    num_chars = len(x['chars'])

    f = os.path.join(d, 'data.bin')
    x = np.fromfile(f, np.uint8)
    z = 4 + 4 + h * w
    x8 = x.reshape(n, z)
    x32 = x.view(np.int32).reshape(n, -1)
    font_ids = x32[:, 0]
    char_ids = x32[:, 1]
    images = x8[:, 8:].reshape(n, 1, h, w)

    plot_id_freqs(font_ids, args.out_font_freqs)
    plot_id_freqs(char_ids, args.out_char_freqs)
    heatmap_images(images, args.out_heat_txt, args.out_heat_img)

    print(font_ids.shape, font_ids.dtype)
    print(char_ids.shape, char_ids.dtype)
    print(images.shape, images.dtype)


if __name__ == '__main__':
    main(parse_args())
