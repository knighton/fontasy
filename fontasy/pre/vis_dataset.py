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
    x.add_argument('--out_font_freqs', type=str, default='')
    x.add_argument('--out_char_freqs', type=str, default='')
    x.add_argument('--out_char_table', type=str, default='')
    x.add_argument('--out_heatmap_txt', type=str, default='')
    x.add_argument('--out_heatmap_img', type=str, default='')
    x.add_argument('--out_heatmap_img_log10', type=str, default='')
    return x.parse_args()


def plot_id_freqs(ids, f):
    x = Counter(ids)
    x = sorted(x.values())
    plt.plot(x)
    plt.savefig(f)
    plt.clf()


def plot_char_table(chars, char_ids, f):
    counts = np.zeros(128, np.int64)
    for i in char_ids:
        c = chars[i]
        counts[c] += 1
    max_count = counts.max()
    out = open(f, 'w')
    for i in range(128):
        txt = chr(i) if 33 <= i < 127 else '.'
        count = counts[i]
        pct = 100 * count / max_count
        end = '\n' if i % 8 == 7 else ' '
        s = '%5.1f %s%s' % (pct, txt, end)
        out.write(s)
    out.close()


def plot_heatmap(images, f_txt, f_img, f_img_log10):
    heatmap = images.astype(np.uint64).sum(0).sum(0)
    h, w = heatmap.shape

    out = open(f_txt, 'w')
    for hh in range(h):
        ss = []
        for ww in range(w):
            count = heatmap[hh, ww] // 1000
            s = '%6d' % count
            ss.append(s)
        line = ' '.join(ss) + '\n'
        out.write(line)
    out.close()

    x = 255 * heatmap / heatmap.max()
    x = x.astype(np.uint8)
    im = Image.fromarray(x)
    im.save(f_img)

    heatmap = np.log10(heatmap + 1)
    x = 255 * heatmap / heatmap.max()
    x = x.astype(np.uint8)
    im = Image.fromarray(x)
    im.save(f_img_log10)


def main(args):
    d = getattr(args, 'in')

    f = os.path.join(d, 'meta.json')
    x = json.load(open(f))
    n = x['img_count']
    h = x['img_height']
    w = x['img_width']
    num_fonts = len(x['fonts'])
    chars = np.array(x['chars'])
    num_chars = len(chars)

    f = os.path.join(d, 'all.bin')
    x = np.fromfile(f, np.uint8)
    z = 4 + 4 + h * w
    x8 = x.reshape(n, z)
    x32 = x.view(np.int32).reshape(n, -1)
    font_ids = x32[:, 0]
    char_ids = x32[:, 1]
    images = x8[:, 8:].reshape(n, 1, h, w)

    plot_id_freqs(font_ids, args.out_font_freqs)
    plot_id_freqs(char_ids, args.out_char_freqs)
    plot_char_table(chars, char_ids, args.out_char_table)
    plot_heatmap(images, args.out_heatmap_txt, args.out_heatmap_img,
                 args.out_heatmap_img_log10)


if __name__ == '__main__':
    main(parse_args())
