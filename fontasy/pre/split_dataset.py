from argparse import ArgumentParser
import json
import numpy as np
import os


def parse_args():
    x = ArgumentParser()
    x.add_argument('--dataset', type=str, required=True)
    x.add_argument('--val_frac', type=float, default=0.1)
    return x.parse_args()


def main(args):
    assert 0 < args.val_frac < 1

    f = os.path.join(args.dataset, 'meta.json')
    x = json.load(open(f))
    c = 1
    h = x['img_height']
    w = x['img_width']
    z = 4 + 4 + c * h * w

    f = os.path.join(args.dataset, 'all.bin')
    in_all = open(f, 'rb')
    f = os.path.join(args.dataset, 'train.bin')
    out_train = open(f, 'wb')
    f = os.path.join(args.dataset, 'val.bin')
    out_val = open(f, 'wb')
    while in_all.peek():
        b = in_all.read(z)
        assert len(b) == z
        out = out_train if args.val_frac < np.random.random() else out_val
        out.write(b)
    out_train.close()
    out_val.close()


if __name__ == '__main__':
    main(parse_args())
