from argparse import Namespace
import numpy as np
from PIL import Image
import json
import os
from shutil import rmtree
import torch
from torch.nn import Parameter as P
from torch.optim import Adam
from torch.optim import Adam
from tqdm import tqdm

from .model import Generator, Planner


class Trainer(object):
    args_fn = 'args.json'

    fonts_fn = 'fonts.f32'
    chars_fn = 'chars.f32'
    planner_fn = 'planner.pt'
    generator_fn = 'generator.pt'
    optimizer_fn = 'optimizer.pt'

    train_losses_fn = 'train_losses.f32'
    val_losses_fn = 'val_losses.f32'
    demo_targets_fn = 'demo_targets.u8'
    demo_preds_fn = 'demo_preds.u8'

    demo_fn = 'demo.png'

    @classmethod
    def from_args(cls, args, num_fonts, num_chars, device):
        fonts = P(torch.randn(num_fonts, args.font_vec_dim).to(device))
        chars = P(torch.randn(num_chars, args.char_vec_dim).to(device))
        planner = Planner(args.font_vec_dim + args.char_vec_dim,
                          args.plan_vec_dim).to(device)
        generator = Generator(args.plan_vec_dim, args.gen_channels).to(device)

        parameters = [fonts, chars] + list(planner.parameters()) + \
            list(generator.parameters())
        optimizer = Adam(parameters)

        models = fonts, chars, planner, generator, optimizer

        return cls(args, models)

    @classmethod
    def args_from_dir(cls, dirname):
        f = os.path.join(dirname, cls.args_fn)
        x = json.load(open(f))
        return Namespace(**x)

    @classmethod
    def models_from_dir(cls, dirname, args, device):
        f = os.path.join(dirname, cls.fonts_fn)
        x = np.fromfile(f, np.float32)
        x = x.reshape(-1, args.font_vec_dim)
        fonts = P(torch.tensor(x, device=device))

        f = os.path.join(dirname, cls.chars_fn)
        x = np.fromfile(f, np.float32)
        x = x.reshape(-1, args.char_vec_dim)
        chars = P(torch.tensor(x, device=device))

        f = os.path.join(dirname, cls.planner_fn)
        planner = Planner(args.font_vec_dim + args.char_vec_dim,
                          args.plan_vec_dim)
        planner.load_state_dict(torch.load(f))
        planner.to(device)

        f = os.path.join(dirname, cls.generator_fn)
        generator = Generator(args.plan_vec_dim, args.gen_channels)
        generator.load_state_dict(torch.load(f))
        generator.to(device)

        f = os.path.join(dirname, cls.optimizer_fn)
        parameters = [fonts, chars] + list(planner.parameters()) + \
            list(generator.parameters())
        optimizer = Adam(parameters)
        optimizer.load_state_dict(torch.load(f))

        return fonts, chars, planner, generator, optimizer

    @classmethod
    def results_from_dir(cls, dirname, args, img_height, img_width):
        f = os.path.join(dirname, cls.train_losses_fn)
        x = np.fromfile(f, np.float32)
        x = x.reshape(-1, args.rounds_per_epoch, args.trains_per_round)
        train_losses = list(x)

        f = os.path.join(dirname, cls.val_losses_fn)
        x = np.fromfile(f, np.float32)
        x = x.reshape(-1, args.rounds_per_epoch, args.vals_per_round)
        val_losses = list(x)

        f = os.path.join(dirname, cls.demo_targets_fn)
        x = np.fromfile(f, np.uint8)
        x = x.reshape(-1, args.demo_size, 1, img_height, img_width)
        demo_targets = list(x)

        f = os.path.join(dirname, cls.demo_preds_fn)
        x = np.fromfile(f, np.uint8)
        x = x.reshape(-1, args.demo_size, 1, img_height, img_width)
        demo_preds = list(x)

        return train_losses, val_losses, demo_targets, demo_preds

    @classmethod
    def from_dir(cls, dirname, device, img_height, img_width):
        args = cls.args_from_dir(dirname)
        models = cls.models_from_dir(dirname, args, device)
        results = cls.results_from_dir(dirname, args, img_height, img_width)
        return cls(args, models, results)

    def __init__(self, args, models, results=None):
        results = results or [], [], [], []

        self.args = args
        self.models = models
        self.results = results

        self.fonts, self.chars, self.planner, self.generator, \
            self.optimizer = models

        self.train_losses, self.val_losses, self.demo_targets, \
            self.demo_preds = results

        self.device = self.fonts.device

    def args_to_dir(self, dirname):
        f = os.path.join(dirname, self.args_fn)
        json.dump(self.args.__dict__, open(f, 'w'))

    def models_to_dir(self, dirname):
        f = os.path.join(dirname, self.fonts_fn)
        self.fonts.detach().cpu().numpy().tofile(f)

        f = os.path.join(dirname, self.chars_fn)
        self.chars.detach().cpu().numpy().tofile(f)

        f = os.path.join(dirname, self.planner_fn)
        torch.save(self.planner.state_dict(), f)

        f = os.path.join(dirname, self.generator_fn)
        torch.save(self.generator.state_dict(), f)

        f = os.path.join(dirname, self.optimizer_fn)
        torch.save(self.optimizer.state_dict(), f)

    def results_to_dir(self, dirname):
        f = os.path.join(dirname, self.train_losses_fn)
        x = np.stack(self.train_losses).tofile(f)

        f = os.path.join(dirname, self.val_losses_fn)
        x = np.stack(self.val_losses).tofile(f)

        f = os.path.join(dirname, self.demo_targets_fn)
        x = np.stack(self.demo_targets).tofile(f)

        f = os.path.join(dirname, self.demo_preds_fn)
        x = np.stack(self.demo_preds).tofile(f)

    def demo_to_dir(self, dirname):
        targets = np.stack(self.demo_targets)
        preds = np.stack(self.demo_preds)
        x = np.stack([targets, preds])
        s, e, n, c, h, w = x.shape
        x = x.transpose(0, 1, 4, 2, 5, 3)
        x = x.reshape(s * e * h, n * w, c)
        assert x.shape[2] == 1
        x = x[:, :, 0]
        im = Image.fromarray(x, 'L')
        f = os.path.join(dirname, self.demo_fn)
        im.save(f)

    def to_dir(self, dirname):
        if os.path.exists(dirname):
            rmtree(dirname)
        os.makedirs(dirname)
        self.args_to_dir(dirname)
        self.models_to_dir(dirname)
        self.results_to_dir(dirname)
        self.demo_to_dir(dirname)

    def get_features(self, font_ids, char_ids):
        fonts = self.fonts[font_ids]
        chars = self.chars[char_ids]
        return torch.cat([fonts, chars], 1)

    def forward(self, features):
        plans = self.planner(features)
        return self.generator(plans)

    def getrain_loss(self, preds, targets):
        return ((preds - targets) ** 2).mean()

    def get_demo_images(self, dataset, num_samples):
        self.planner.eval()
        self.generator.eval()
        with torch.no_grad():
            targets, font_ids, char_ids = dataset.get_batch(
                True, num_samples, self.device)
            features = self.get_features(font_ids, char_ids)
            preds = self.forward(features)
            targets = (255 * targets).type(torch.uint8).cpu().numpy()
            preds = (255 * preds).type(torch.uint8).cpu().numpy()
        return targets, preds

    def train_on_batch(self, dataset):
        self.optimizer.zero_grad()
        targets, font_ids, char_ids = dataset.get_batch(
            True, self.args.batch_size, self.device)
        features = self.get_features(font_ids, char_ids)
        preds = self.forward(features)
        loss = self.getrain_loss(preds, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_on_batch(self, dataset):
        with torch.no_grad():
            targets, font_ids, char_ids = dataset.get_batch(
                False, self.args.batch_size, self.device)
            features = self.get_features(font_ids, char_ids)
            preds = self.forward(features)
            loss = self.getrain_loss(preds, targets)
        return loss.item()

    def fit_round(self, dataset):
        train_losses = np.zeros(self.args.trains_per_round, np.float32)
        self.planner.train()
        self.generator.train()
        for batch_id in range(self.args.trains_per_round):
            train_losses[batch_id] = self.train_on_batch(dataset)
        val_losses = np.zeros(self.args.vals_per_round, np.float32)
        self.planner.eval()
        self.generator.eval()
        for batch_id in range(self.args.vals_per_round):
            val_losses[batch_id] = self.val_on_batch(dataset)
        return train_losses, val_losses

    def fit_epoch(self, dataset):
        epoch_id = len(self.train_losses)

        train_shape = self.args.rounds_per_epoch, self.args.trains_per_round
        train_losses = np.zeros(train_shape, np.float32)
        val_shape = self.args.rounds_per_epoch, self.args.vals_per_round
        val_losses = np.zeros(val_shape, np.float32)
        total = self.args.rounds_per_epoch
        for round_id in tqdm(range(total), total=total, leave=False):
            trains, vals = self.fit_round(dataset)
            train_losses[round_id] = trains
            val_losses[round_id] = vals
        self.train_losses.append(train_losses)
        self.val_losses.append(val_losses)

        demo_targets, demo_preds = self.get_demo_images(
            dataset, self.args.demo_size)
        self.demo_targets.append(demo_targets)
        self.demo_preds.append(demo_preds)

        line = '%6d %6.4f %6.4f' % (epoch_id, train_losses.mean(),
                                    val_losses.mean())
        print(line)

    def fit(self, dataset, dirname):
        for epoch_id in range(len(self.train_losses), self.args.num_epochs):
            self.fit_epoch(dataset)
            if dirname:
                self.to_dir(dirname)
