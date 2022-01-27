import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataset import make_data_loader
from models import get_model
import config as cfg
from losses import Loss
from utils.utils import save_checkpoint, non_max_suppression, get_batch_statistics, ap_per_class, xywh2xyxy, weights_init_normal


os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_ids
log_writer = SummaryWriter()


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # Define Dataloader
        self.train_loader, self.val_loader, _, self.nclass = make_data_loader(self.cfg)

        self.model = get_model(self.cfg.name, self.cfg.model_config, self.cfg)
        self.model.apply(weights_init_normal)

        # If specified we start from checkpoint
        if self.cfg.pretrained_model:
            if self.cfg.pretrained_model.endswith(".pth"):
                self.model.load_state_dict(torch.load(self.cfg.pretrained_model))
                print('---------------------Pretrained model is loaded!---------------------')

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum,
                                         weight_decay=self.cfg.weight_decay, nesterov=self.cfg.nesterov)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8,
                                                                    patience=8, verbose=True, min_lr=0.00001)
        if self.cfg.cuda:
            self.model.cuda()

        self.best_AP = 0.0

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        train_loss = 0.0
        acc = 0.0

        for i, sample in enumerate(tbar):
            _, img, target = sample
            if self.cfg.cuda:
                img, target = img.cuda(), target.cuda()

            self.optimizer.zero_grad()
            loss, output = self.model(img, target)

            # loss = self.criterion(output, target, weight)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            log_writer.add_scalar('Train/loss', float(train_loss / (i + 1)), epoch)


        # log_writer.add_scalar('Train/acc', float(acc / (len(self.train_loader)*self.cfg.batch_size)), epoch)

        # print('Train accuracy: {}'.format(acc / (len(self.train_loader)*self.cfg.batch_size)))
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.batch_size + img.data.shape[0]))

    def validate(self, iou_thres, conf_thres, nms_thres):
        self.model.eval()
        tbar = tqdm(self.val_loader)

        sample_metrics = []
        labels = []
        for i, sample in enumerate(tbar):
            _, img, target = sample

            if target is None:
                continue

            labels += target[:, 1].tolist()
            # Rescale target
            target[:, 2:] = xywh2xyxy(target[:, 2:])
            target[:, 2:] *= cfg.input_size

            if self.cfg.cuda:
                img = img.cuda()

            with torch.no_grad():
                outputs = self.model(img)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            sample_metrics += get_batch_statistics(outputs, target, iou_threshold=iou_thres)

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        print(f"---- mAP {AP.mean()}")
        if AP.mean() > self.best_AP:
            torch.save(self.model.state_dict(), '{}_{:.2f}.pth'.format(self.cfg.model_path.split('.')[0], AP.mean()))
            self.best_AP = AP.mean()
            print('save model to {}'.format('{}_{:.2f}.pth'.format(self.cfg.model_path.split('.')[0], AP.mean())))

        return precision, recall, AP, f1, ap_class


def train():
    if cfg.model_path is None:
        cfg.model_path = 'weights/detection_{}.pth'.format(datetime.datetime.now().strftime("%Y-%m-%d"))
    trainer = Trainer(cfg)
    print('Starting Epoch:', trainer.cfg.begin_epoch)
    print('Total Epoches:', trainer.cfg.epoch)
    for epoch in range(trainer.cfg.begin_epoch, trainer.cfg.epoch):
        trainer.train(epoch)
        precision, recall, AP, f1, ap_class = trainer.validate(iou_thres=0.5, conf_thres=0.5, nms_thres=0.5)
        trainer.scheduler.step(AP.mean())


if __name__ == '__main__':
    train()