from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from PIL import ImageFont
from queue import Queue
from threading import Thread


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--out', type=str, required=True)
    x.add_argument('--num_threads', type=int, default=64)
    return x.parse_args()


def each_ttf(d):
    for x in sorted(os.listdir(d)):
        x = os.path.join(d, x)
        if os.path.isdir(x):
            for r in each_ttf(x):
                yield r
        elif os.path.isfile(x) and x.lower().endswith('.ttf'):
            yield x


def process(f):
    b = open(f, 'rb').read()
    z = len(b)
    h = sha256(b).hexdigest()
    font = ImageFont.truetype(f, 10)
    a = font.getname()
    return {
        'file': f,
        'size': z,
        'sha256': h,
        'name': a,
    }


def work_thread(filename_q, result_q):
    while True:
        f = filename_q.get()
        x = process(f)
        result_q.put(x)
        filename_q.task_done()


def main(args):
    filename_q = Queue()
    result_q = Queue()

    for i in range(args.num_threads):
        Thread(target=work_thread, args=(filename_q, result_q),
               daemon=True).start()

    in_root = getattr(args, 'in')
    for f in each_ttf(in_root):
        filename_q.put(f)

    filename_q.join()

    out = open(args.out, 'w')
    while not result_q.empty():
        x = result_q.get()
        s = json.dumps(x, sort_keys=True)
        out.write(s + '\n')
        result_q.task_done()
    out.close()


if __name__ == '__main__':
    main(parse_args())
