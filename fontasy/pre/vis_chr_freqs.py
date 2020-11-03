from argparse import ArgumentParser
from collections import Counter, defaultdict
import json


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--out_by_chr', type=str, required=True)
    x.add_argument('--out_by_freq', type=str, required=True)
    x.add_argument('--out_ascii', type=str, required=True)
    return x.parse_args()


def main(args):
    in_f = getattr(args, 'in')
    c2n = Counter()
    total = 0
    for s in open(in_f):
        x = json.loads(s)
        cc = x['chrs']
        for c in cc:
            c2n[c] += 1
        total += 1

    out = open(args.out_by_chr, 'w')
    for c in sorted(c2n):
        n = c2n[c]
        x = {
            'chr': chr(c),
            'ord': c,
            'count': n,
            'frac': n / total,
        }
        s = json.dumps(x, sort_keys=True) + '\n'
        out.write(s)
    out.close()

    n2cc = defaultdict(list)
    for c in sorted(c2n):
        n = c2n[c]
        n2cc[n].append(c)

    out = open(args.out_by_freq, 'w')
    for n in sorted(n2cc, reverse=True):
        for c in n2cc[n]:
            x = {
                'chr': chr(c),
                'ord': c,
                'count': n,
                'frac': n / total,
            }
            s = json.dumps(x, sort_keys=True) + '\n'
            out.write(s)
    out.close()

    out = open(args.out_ascii, 'w')
    for row in range(256 // 8):
        ss = []
        for col in range(8):
            c = row * 8 + col
            n = c2n[c]
            pct = 100 * n / total
            txt = chr(c) if 33 <= c < 127 or 160 <= c else ' '
            s = '%5.1f%% %s' % (pct, txt)
            ss.append(s)
        s = '  '.join(ss) + '\n'
        out.write(s)
    out.close()


if __name__ == '__main__':
    main(parse_args())
