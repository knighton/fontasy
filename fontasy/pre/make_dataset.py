from argparse import ArgumentParser
from fontTools.ttLib import TTFont
import json
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import struct
from tqdm import tqdm
from wurlitzer import pipes


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--chars', type=str, required=True)
    x.add_argument('--font_size', type=int, required=True)
    x.add_argument('--max_ascent', type=int, required=True)
    x.add_argument('--max_descent', type=int, required=True)
    x.add_argument('--img_width', type=int, required=True)
    x.add_argument('--out', type=str, required=True)
    return x.parse_args()


def parse_char_ranges(s):
    ss = s.split(',')
    rr = []
    for s in ss:
        if '..' in s:
            a, b = s.split('..')
            a = int(a)
            b = int(b)
            assert a <= b
            rr += list(range(a, b + 1))
        else:
            r = int(s)
            rr.append(r)
    return sorted(rr)


def is_round(x):
    return 2 ** int(np.log2(x)) == x


def is_font_height_ok(font, max_ascent, max_descent):
    a, d = font.getmetrics()
    assert 1 <= a
    d = abs(d)
    assert 0 <= d
    if max_ascent < a:
        return False
    if max_descent < d:
        return False
    return True


def get_font_chars(f):
    font = TTFont(f)
    r = set()
    with pipes() as (out, err):
        for x in font['cmap'].tables:
            r |= set(x.cmap)
    return r


def center_char_in_img(a, w):
    _, a_w = a.shape
    assert (a[:, w:] == 0).all()
    pad_lr = a_w - w
    pad_left = pad_lr // 2
    pad_right = pad_lr - pad_left
    x = np.zeros(a.shape, a.dtype)
    x[:, pad_left:-pad_right] = a[:, :-pad_lr]
    return x
    

def sample_to_bytes(font_id, char_id, image):
    return struct.pack('ii', font_id, char_id) + image.tobytes()


def main(args):
    os.makedirs(args.out)

    want_chars = parse_char_ranges(args.chars)
    want_chars_set = set(want_chars)
    c2id = {}
    for c in want_chars:
        c2id[c] = len(c2id)

    img_height = args.max_ascent + args.max_descent
    assert is_round(img_height)

    in_f = getattr(args, 'in')
    lines = open(in_f).readlines()
    lines = tqdm(lines, leave=False)

    f = os.path.join(args.out, 'data.bin')
    out = open(f, 'wb')
    bytes_per_sample = 4 + 4 + img_height * args.img_width
    font_names = []
    img_count = 0
    for font_id, line in enumerate(lines):
        x = json.loads(line)
        f = x['file']
        font_names.append(x['name'])
        font = ImageFont.truetype(f, args.font_size)
        if not is_font_height_ok(font, args.max_ascent, args.max_descent):
            continue
        have_chars_set = get_font_chars(f)
        cc = sorted(want_chars_set & have_chars_set)
        for c in cc:
            char_id = c2id[c]
            text = chr(c)
            image = Image.new('L', (args.img_width, img_height))
            draw = ImageDraw.Draw(image)
            w, h = draw.textsize(text, font=font)
            if img_height < h or args.img_width < w:
                continue
            draw.text((0, 0), text, font=font, fill=(255,))
            a = np.array(image)
            a = center_char_in_img(a, w)
            b = sample_to_bytes(font_id, char_id, a)
            assert len(b) == bytes_per_sample
            out.write(b)
            img_count += 1
    out.close()

    f = os.path.join(args.out, 'meta.json')
    out = open(f, 'w')
    x = {
        'fonts': font_names,
        'chars': want_chars,
        'font_size': args.font_size,
        'max_ascent': args.max_ascent,
        'max_descent': args.max_descent,
        'img_count': img_count,
        'img_height': img_height,
        'img_width': args.img_width,
    } 
    s = json.dumps(x, sort_keys=True)
    out.write(s)

    f = os.path.join(args.out, 'meta.bin')
    out = open(f, 'wb')
    a = struct.pack('iii', img_count, img_height, args.img_width)
    b = struct.pack('iii', args.font_size, args.max_ascent, args.max_descent)
    c = struct.pack('ii', len(want_chars), len(font_names))
    d = np.array(want_chars, np.int32).tobytes()
    out.write(a + b + c + d)
    for font_name in font_names:
        s = ' '.join(font_name) + '\n'
        out.write(s.encode('utf-8'))
    out.close()


if __name__ == '__main__':
    main(parse_args())
