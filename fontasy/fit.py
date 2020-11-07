from argparse import ArgumentParser
import os
import torch

from .dataset import Dataset
from .trainer import Trainer


def parse_args():
    x = ArgumentParser()
    x.add_argument('--dataset', type=str, required=True)
    x.add_argument('--val_frac', type=float, default=0.2)
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--font_vec_dim', type=int, default=64)
    x.add_argument('--char_vec_dim', type=int, default=64)
    x.add_argument('--plan_vec_dim', type=int, default=64)
    x.add_argument('--gen_channels', type=int, default=32)
    x.add_argument('--num_epochs', type=int, default=10000)
    x.add_argument('--rounds_per_epoch', type=int, default=100)
    x.add_argument('--trains_per_round', type=int, default=10)
    x.add_argument('--vals_per_round', type=int, default=1)
    x.add_argument('--batch_size', type=int, default=64)
    x.add_argument('--demo_size', type=int, default=32)
    x.add_argument('--save', type=str, required=True)
    return x.parse_args()


def main(args):
    assert not os.path.exists(args.save)
    device = torch.device(args.device)
    dataset = Dataset.from_dir(args.dataset, args.val_frac)
    trainer = Trainer.from_args(args, dataset.num_fonts, dataset.num_chars, device)
    trainer.fit(dataset, args.save)


if __name__ == '__main__':
    main(parse_args())
