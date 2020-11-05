import json
import numpy as np
import os
import torch


class Dataset(object):
    @classmethod
    def from_dir(cls, d, val_frac):
        assert 0 < val_frac < 1

        f = os.path.join(d, 'meta.json')
        x = json.load(open(f))
        n = x['img_count']
        h = x['img_height']
        w = x['img_width']
        z = 4 + 4 + h * w

        f = os.path.join(d, 'data.bin')
        x = np.fromfile(f, np.uint8)
        x8 = x.reshape(n, z)
        images = x8[:, 8:].reshape(n, 1, h, w)
        x32 = x.view(np.int32).reshape(n, -1)
        font_ids = x32[:, 0]
        char_ids = x32[:, 1]

        num_fonts = int(font_ids.max()) + 1
        prob = val_frac, 1 - val_frac
        font_splits = np.random.choice(2, num_fonts, p=prob)

        return cls(font_ids, char_ids, images, font_splits)

    def __init__(self, font_ids, char_ids, images, font_splits):
        self.sample_font_ids = font_ids
        self.sample_char_ids = char_ids
        self.sample_images = images
        self.font_splits = font_splits

        self.num_fonts = int(font_ids.max()) + 1
        self.num_chars = int(char_ids.max()) + 1

        train_sample_ids = []
        val_sample_ids = []
        for sample_id, font_id in enumerate(font_ids):
            split = font_splits[font_id]
            sample_ids = train_sample_ids if split else val_sample_ids
            sample_ids.append(sample_id)
        self.train_sample_ids = np.array(train_sample_ids, np.int32)
        self.val_sample_ids = np.array(val_sample_ids, np.int32)

    def get_batch(self, training, size, device):
        split_sample_ids = self.train_sample_ids if training else \
                           self.val_sample_ids
        sample_ids = np.random.choice(split_sample_ids, size)
        images = self.sample_images[sample_ids]
        images = torch.tensor(images, dtype=torch.float32, device=device)
        images = images / 255
        font_ids = self.sample_font_ids[sample_ids]
        font_ids = torch.tensor(font_ids, dtype=torch.int64, device=device)
        char_ids = self.sample_char_ids[sample_ids]
        char_ids = torch.tensor(char_ids, dtype=torch.int64, device=device)
        return images, font_ids, char_ids
