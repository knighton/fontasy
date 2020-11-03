from argparse import ArgumentParser
from collections import Counter, defaultdict
import json


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--font_prop', type=str, required=True)
    x.add_argument('--out', type=str, required=True)
    return x.parse_args()


def each_line(file_name, prop):
    for line in open(file_name):
        x = json.loads(line)
        _, got_prop = x['name']
        got_prop = got_prop.lower()
        if got_prop == prop:
            yield line


def main(args):
    assert args.font_prop.islower()
    in_f = getattr(args, 'in')
    out = open(args.out, 'w')
    for line in each_line(in_f, args.font_prop):
        out.write(line)
    out.close()


if __name__ == '__main__':
    main(parse_args())
