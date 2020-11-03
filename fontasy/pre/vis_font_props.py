from argparse import ArgumentParser
from collections import Counter, defaultdict
import json


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--out', type=str, required=True)
    return x.parse_args()


def main(args):
    prop2count = Counter()
    in_f = getattr(args, 'in')
    for s in open(in_f):
        x = json.loads(s)
        family, prop = x['name']
        p = prop.lower()
        prop2count[p] += 1

    count2props = defaultdict(list)
    for prop in sorted(prop2count):
        count = prop2count[prop]
        count2props[count].append(prop)

    out = open(args.out, 'w')
    for count in sorted(count2props, reverse=True):
        for prop in count2props[count]:
            line = '%7d %s\n' % (count, prop)
            out.write(line)
    out.close()


if __name__ == '__main__':
    main(parse_args())
