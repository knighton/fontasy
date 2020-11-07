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

        prob = val_frac, 1 - val_frac
        splits = np.random.choice(2, n, p=prob)

        f = os.path.join(d, 'data.bin')
        x = np.fromfile(f, np.uint8)
        x8 = x.reshape(n, z)
        images = x8[:, 8:].reshape(n, 1, h, w)
        x32 = x.view(np.int32).reshape(n, -1)
        font_ids = x32[:, 0]
        char_ids = x32[:, 1]

        return cls(splits, font_ids, char_ids, images)

    def __init__(self, splits, font_ids, char_ids, images):
        self.sample_splits = splits
        self.sample_font_ids = font_ids
        self.sample_char_ids = char_ids
        self.sample_images = images

        self.num_fonts = int(font_ids.max()) + 1
        self.num_chars = int(char_ids.max()) + 1

        self.num_samples, self.img_channels, self.img_height, \
            self.img_width = images.shape

        tt = []
        vv = []
        for sample_id, training in enumerate(splits):
            sample_ids = tt if training else vv
            sample_ids.append(sample_id)
        self.train_sample_ids = np.array(tt, np.int32)
        self.val_sample_ids = np.array(vv, np.int32)

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
