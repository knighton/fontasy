from argparse import ArgumentParser
from fontTools.ttLib import TTFont
import json
from wurlitzer import pipes


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--out', type=str, required=True)
    return x.parse_args()


def get_font_chrs(f):
    font = TTFont(f)
    r = set()
    with pipes() as (out, err):
        for x in font['cmap'].tables:
            r |= set(x.cmap)
    return sorted(r)


def main(args):
    in_f = getattr(args, 'in')
    out = open(args.out, 'w')
    for s in open(in_f):
        x = json.loads(s)
        f = x['file']
        h = x['sha256']
        cc = get_font_chrs(f)
        x = {
            'sha256': h,
            'chrs': cc,
        }
        s = json.dumps(x, sort_keys=True) + '\n'
        out.write(s)
    out.close()


if __name__ == '__main__':
    main(parse_args())
