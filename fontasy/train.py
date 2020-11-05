from argparse import ArgumentParser
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import Parameter as P
import os
from torch.optim import Adam
from tqdm import tqdm

from .dataset import Dataset
from .model import Designer, Generator


def parse_args():
    x = ArgumentParser()
    x.add_argument('--dataset', type=str, required=True)
    x.add_argument('--val_frac', type=float, default=0.2)
    x.add_argument('--out', type=str, default='data/out/')
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--font_dim', type=int, default=64)
    x.add_argument('--char_dim', type=int, default=64)
    x.add_argument('--design_dim', type=int, default=64)
    x.add_argument('--gen_channels', type=int, default=32)
    x.add_argument('--num_epochs', type=int, default=10000)
    x.add_argument('--rounds_per_epoch', type=int, default=100)
    x.add_argument('--trains_per_round', type=int, default=10)
    x.add_argument('--vals_per_round', type=int, default=1)
    x.add_argument('--batch_size', type=int, default=128)
    return x.parse_args()


def save_batch(out_root, epoch_id, true_images, pred_images):
    x = torch.stack([true_images, pred_images])
    x = x.squeeze(2)
    nh, nw, h, w = x.shape
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(nh * h, nw * w)
    x = (255 * x).type(torch.uint8)
    x = x.cpu().numpy()
    im = Image.fromarray(x)
    f = os.path.join(out_root, 'epoch_%06d.png' % epoch_id)
    im.save(f)


def get_loss(pred, true):
    return ((pred - true) ** 2).mean()


def main(args):
    os.makedirs(args.out)

    device = torch.device(args.device)

    dataset = Dataset.from_dir(args.dataset, args.val_frac)

    font_embeds = P(torch.randn(dataset.num_fonts, args.font_dim).to(device))
    char_embeds = P(torch.randn(dataset.num_chars, args.char_dim).to(device))
    designer_in_dim = args.font_dim + args.char_dim
    designer = Designer(designer_in_dim, args.design_dim).to(device)
    generator = Generator(args.design_dim, args.gen_channels).to(device)
    params = [font_embeds, char_embeds] + list(designer.parameters()) + \
        list(generator.parameters())
    optimizer = Adam(params)

    def get_batch(training, batch_size):
        images, font_ids, char_ids = dataset.get_batch(
            training, batch_size, device)
        fonts = font_embeds[font_ids]
        chars = char_embeds[char_ids]
        features = torch.cat([fonts, chars], 1)
        return features, images

    for epoch_id in range(args.num_epochs):
        t_losses = []
        v_losses = []

        for round_id in tqdm(range(args.rounds_per_epoch),
                             total=args.rounds_per_epoch, leave=False):
            designer.train()
            generator.train()
            for train_id in range(args.trains_per_round):
                optimizer.zero_grad()
                features, true_images = get_batch(True, args.batch_size)
                designs = designer(features)
                pred_images = generator(designs)

                if not round_id and not train_id:
                    save_batch(args.out, epoch_id, true_images, pred_images)

                loss = get_loss(pred_images, true_images)
                loss.backward()
                optimizer.step()
                t_losses.append(loss.item())

            with torch.no_grad():
                designer.eval()
                generator.eval()
                for val_id in range(args.vals_per_round):
                    features, true_images = get_batch(True, args.batch_size)
                    designs = designer(features)
                    pred_imags = generator(designs)
                    loss = get_loss(pred_images, true_images)
                    v_losses.append(loss.item())

        t_loss = np.mean(t_losses)
        v_loss = np.mean(v_losses)
        print('%7d %7.4f %7.4f' % (epoch_id, t_loss, v_loss))


if __name__ == '__main__':
    main(parse_args())
