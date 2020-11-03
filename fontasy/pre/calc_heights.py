from argparse import ArgumentParser
import json
import numpy as np
from PIL import ImageFont


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--min_font_size', type=int, required=True)
    x.add_argument('--max_font_size', type=int, required=True)
    x.add_argument('--out', type=str, required=True)
    return x.parse_args()


def fix(x):
    ascent = x[:, 0]
    assert (0 < ascent).all()
    assert (np.sort(ascent) == ascent).all()

    descent = x[:, 1]
    sign_of_biggest = np.sign(descent[-1])
    descent = sign_of_biggest * descent 
    assert (0 <= descent).all()
    assert (np.sort(descent) == descent).all()

    return np.stack([ascent, descent], 1)


def main(args):
    in_f = getattr(args, 'in')
    assert 1 <= args.min_font_size <= args.max_font_size
    out = open(args.out, 'wb')
    for line in open(in_f):
        x = json.loads(line)
        f = x['file']
        rr = []
        for size in range(args.min_font_size, args.max_font_size + 1):
            font = ImageFont.truetype(f, size)
            r = font.getmetrics()
            rr.append(r)
        x = np.array(rr, np.int16)
        x = fix(x)
        out.write(x.tobytes())
    out.close()


if __name__ == '__main__':
    main(parse_args())
