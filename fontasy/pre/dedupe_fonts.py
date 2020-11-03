from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--out', type=str, required=True)
    x.add_argument('--out_hash2files', type=str, default='')
    x.add_argument('--out_name2files', type=str, default='')
    return x.parse_args()


def each_deduped_font_line(f, hash2ff, name2ff):
    for line in open(f):
        x = json.loads(line)
        f = x['file']
        h = x['sha256']
        if h in hash2ff:
            hash2ff[h].append(f)
            continue
        hash2ff[h].append(f)
        a = tuple(x['name'])
        if a in name2ff:
            name2ff[a].append(f)
            continue
        name2ff[a].append(f)
        yield line


def save_dict(d, f, key, value):
    assert key != value
    out = open(f, 'w')
    for k in sorted(d):
        v = d[k]
        x = {
            key: k,
            value: v,
        }
        s = json.dumps(x, sort_keys=True)
        out.write(s + '\n')
    out.close()


def main(args):
    in_f = getattr(args, 'in')
    hash2ff = defaultdict(list)
    name2ff = defaultdict(list)

    out = open(args.out, 'w')
    for line in each_deduped_font_line(in_f, hash2ff, name2ff):
        out.write(line)
    out.close()

    if args.out_hash2files:
        save_dict(hash2ff, args.out_hash2files, 'sha256', 'files')

    if args.out_name2files:
        save_dict(name2ff, args.out_name2files, 'name', 'files')


if __name__ == '__main__':
    main(parse_args())
