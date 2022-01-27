from dataset.datasets import ListDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch

from dataset.utils import AUGMENTATION_TRANSFORMS, DEFAULT_TRANSFORMS


def make_data_loader(cfg):
    train_set = ListDataset(cfg, cfg.train_file, transform=AUGMENTATION_TRANSFORMS)
    val_set = ListDataset(cfg, cfg.val_file, transform=DEFAULT_TRANSFORMS)
    test_set = ListDataset(cfg, cfg.test_file, transform=DEFAULT_TRANSFORMS)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.workers,
        pin_memory=True, collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=1, collate_fn=val_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=1, collate_fn=test_set.collate_fn)
    num_class = cfg.num_classes

    return train_loader, val_loader, test_loader, num_class