from collections import Counter
import json
import numpy as np
import os
import torch


class Dataset(object):
    @classmethod
    def load_img_shape(cls, dirname):
        f = os.path.join(dirname, 'meta.json')
        x = json.load(open(f))
        c = 1
        h = x['img_height']
        w = x['img_width']
        return c, h, w

    @classmethod
    def load_split(cls, dirname, split, img_shape):
        f = os.path.join(dirname, '%s.bin' % split)
        x = np.fromfile(f, np.uint8)
        c, h, w = img_shape
        z = 4 + 4 + c * h * w
        shape = (-1,) + img_shape
        imgs = x.reshape(-1, z)[:, 8:].reshape(shape)
        n = len(imgs)
        x32 = x.view(np.int32).reshape(n, -1)
        font_ids = x32[:, 0]
        char_ids = x32[:, 1]
        return imgs, font_ids, char_ids

    @classmethod
    def from_dir(cls, dirname):
        img_shape = cls.load_img_shape(dirname)
        train = cls.load_split(dirname, 'train', img_shape)
        val = cls.load_split(dirname, 'val', img_shape)
        return cls(train, val)

    def __init__(self, train, val):
        self.train = t_imgs, t_font_ids, t_char_ids = train
        self.val = v_imgs, v_font_ids, v_char_ids = val

        self.num_train_samples, self.img_channels, self.img_height, \
            self.img_width = t_imgs.shape
        self.num_val_samples = v_imgs.shape[0]
        self.num_samples = self.num_train_samples + self.num_val_samples

        self.num_fonts = max(max(t_font_ids), max(v_font_ids)) + 1
        self.num_chars = max(max(t_char_ids), max(v_char_ids)) + 1

    def get_batch(self, training, size, device):
        imgs, font_ids, char_ids = self.train if training else self.val
        i = np.random.choice(imgs.shape[0], size)
        imgs = torch.tensor(imgs[i], dtype=torch.float32, device=device) / 255
        font_ids = torch.tensor(font_ids[i], dtype=torch.int64, device=device)
        char_ids = torch.tensor(char_ids[i], dtype=torch.int64, device=device)
        return imgs, font_ids, char_ids
