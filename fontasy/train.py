from argparse import ArgumentParser
import torch
from torch import nn
from torch.optim import Adam

from .model import Generator


def parse_args():
    x = ArgumentParser()
    x.add_argument('--dataset', type=str, required=True)
    x.add_argument('--save', type=str, default='/dev/stdout')
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--font_embed_dim', type=int, default=128)
    x.add_argument('--glyph_embed_dim', type=int, default=128)
    x.add_argument('--gen_body_channels', type=int, default=16)
    x.add_argument('--num_epochs', type=int, default=1000)
    x.add_argument('--rounds_per_epoch', type=int, default=100)
    x.add_argument('--trains_per_round', type=int, default=10)
    x.add_argument('--vals_per_round', type=int, default=1)
    return x.parse_args()


def main(args):
    device = torch.device(args.device)

    dataset = load_dataset(args.dataset)

    font_embeds = torch.randn(dataset.num_fonts, args.font_embed_dim,
                              requires_grad=True).to(device)
    glyph_embeds = torch.randn(dataset.glyphs_per_font, args.glyph_embed_dim,
                               requires_grad=True).to(device)
    gen_in_dim = args.font_embed_dim + args.glyph_embed_dim
    generator = Generator(gen_in_dim, args.gen_body_channels).to(device)
    params = [font_embeds, glyph_embeds] + list(generator.parameters())
    optimizer = Adam(params)

    saver = Saver(args.save)

    def get_batch(training, batch_size):
        images, font_ids, glyph_ids = dataset.get(training, batch_size)
        font_embed = font_embeds[font_ids]
        glyph_embed = glyph_embeds[glyph_ids]
        features = torch.cat([font_embed, glyph_embed], 1)
        images = torch.tensor(images, device=device)
        return features, images

    for epoch_id in range(args.num_epochs):
        for round_id in range(args.rounds_per_epoch):
            generator.train()
            for train_id in range(args.trains_per_round):
                optimizer.zero_grad()
                features, true_images = get_batch(True, args.batch_size)
                pred_images = generator(features)
                loss = F.binary_cross_entropy(pred_images, true_images)
                loss.backward()
                optimizer.step()
                acc = (pred_images.round() == true_images.round()).mean()
                saver.on_train_done(loss.item(), acc)

            with torch.no_grad():
                generator.eval()
                for val_id in range(args.vals_per_round):
                    features, true_images = get_batch(True, args.batch_size)
                    pred_images = generator(features)
                    loss = F.binary_cross_entropy(pred_images, true_images)
                    acc = (pred_images.round() == true_images.round()).mean()
                    saver.on_val_done(loss.item(), acc)


if __name__ == '__main__':
    main(parse_args())
