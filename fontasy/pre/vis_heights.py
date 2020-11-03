from argparse import ArgumentParser
import numpy as np


def parse_args():
    x = ArgumentParser()
    x.add_argument('--in', type=str, required=True)
    x.add_argument('--min_font_size', type=int, required=True)
    x.add_argument('--max_font_size', type=int, required=True)
    x.add_argument('--img_height', type=int, required=True)
    x.add_argument('--out_coverage', type=str, required=True)
    x.add_argument('--out_best', type=str, required=True)
    return x.parse_args()


def main(args):
    font_sizes = np.arange(args.min_font_size, args.max_font_size + 1)
    num_font_sizes = len(font_sizes)

    in_f = getattr(args, 'in')
    x = np.fromfile(in_f, np.int16)
    x = x.reshape(-1, num_font_sizes, 2)
    num_fonts = x.shape[0]
    ascents = x[:, :, 0]
    descents = x[:, :, 1]

    assert (1 <= ascents).all()
    assert (0 <= descents).all()
    assert (np.sort(x, 1) == x).all()

    num_aligns = args.img_height + 1
    max_ascents = np.arange(num_aligns)
    max_descents = num_aligns - np.arange(num_aligns) - 1

    assert (max_ascents + max_descents == args.img_height).all()

    ascent_ok = (ascents.reshape(1, num_fonts, num_font_sizes) <=
                 max_ascents.reshape(num_aligns, 1, 1))
    descent_ok = (descents.reshape(1, num_fonts, num_font_sizes) <=
                  max_descents.reshape(num_aligns, 1, 1))
    ok = ascent_ok * descent_ok
    counts = ok.sum(1)

    out = open(args.out_coverage, 'w')
    for align_id in range(num_aligns):
        ss = []
        for font_size_id in range(num_font_sizes):
            font_size = font_sizes[font_size_id]
            count = counts[align_id, font_size_id]
            pct = 100 * count / num_fonts
            s = '%3d' % pct
            ss.append(s)
        line = ' '.join(ss) + '\n'
        out.write(line)
    out.close()

    best_aligns = counts.argmax(0)
    best_counts = np.take_along_axis(counts, np.expand_dims(best_aligns, 0),
                                     0).squeeze(0)

    out = open(args.out_best, 'w')
    out.write('FontSize,MaxAscent,MaxDescent,NumFonts,PctFonts\n')
    for i, (align, count) in enumerate(zip(best_aligns, best_counts)):
        font_size = font_sizes[i]
        max_ascent = max_ascents[align]
        max_descent = max_descents[align]
        pct = 100 * count / num_fonts
        line = '%d,%d,%d,%d,%.3f\n' % \
            (font_size, max_ascent, max_descent, count, pct)
        out.write(line)
    out.close()


if __name__ == '__main__':
    main(parse_args())
