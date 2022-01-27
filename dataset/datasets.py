from torch.utils.data import Dataset
import cv2
import numpy as np
import warnings
import random
import torch
import os

from dataset.utils import resize


class ListDataset(Dataset):
    def __init__(self, cfg, list_path, multiscale=False, transform=None):
        '''
        :param cfg: config
        :param list_path: data file
        :param multiscale: whether scale images to different size randomly
        :param transform:
        '''
        self.cfg = cfg
        # self.root_data_dir = os.path.dirname(list_path)
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]

        self.img_size = self.cfg.input_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        # read image
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        except Exception as e:
            print('Could not read image {}'.format(img_path))
            return

        # read label
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            # ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print('Could not read label {}'.format(label_path))
            return

        # transform image and label
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except:
                print('Could not apply transform')
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]
        paths, imgs, bb_targets = list(zip(*batch))

        # Select new image size every ten batches
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # add sample batch index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)