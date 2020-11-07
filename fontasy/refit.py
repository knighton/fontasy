from argparse import ArgumentParser
import os
import torch

from .dataset import Dataset
from .trainer import Trainer


def parse_args():
    x = ArgumentParser()
    x.add_argument('--dataset', type=str, required=True)
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--save', type=str, required=True)
    return x.parse_args()


def main(args):
    device = torch.device(args.device)
    dataset = Dataset.from_dir(args.dataset)
    trainer = Trainer.from_dir(args.save, device, dataset.img_height,
                               dataset.img_width)
    trainer.fit(dataset, args.save)


if __name__ == '__main__':
    main(parse_args())
