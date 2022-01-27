import torch
import os
import cv2
import numpy as np
# import imgaug.augmenters as iaa
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
# import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import make_data_loader
from models import get_model
import config as cfg
from utils.utils import non_max_suppression, get_batch_statistics, ap_per_class, xywh2xyxy, rescale_boxes, load_classes,\
                        pad_resize

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_ids


class Test:
    def __init__(self, weights_path, cfg):
        self.cfg = cfg

        # Define Dataloader
        self.train_loader, _, self.test_loader,  self.nclass = make_data_loader(self.cfg)

        self.model = get_model(self.cfg.name, self.cfg.model_config, self.cfg)

        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(weights_path))
        # checkpoint = torch.load(weights_path)
        # self.model.load_state_dict(checkpoint)
        print('{}model weight is loaded!{}'.format('-'*30, '-'*30))
        if self.cfg.cuda:
            self.model.cuda()
        # self.augmentations = iaa.Sequential([
        #     iaa.PadToAspectRatio(
        #         1.0,
        #         position="center-center").to_deterministic()
        # ])
        self.classes = load_classes(self.cfg.classes_path)

    def evaluate(self, iou_thres, conf_thres, nms_thres):
        self.model.eval()
        tbar = tqdm(self.test_loader)

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

    def inference(self, img_path, res_dir):
        self.model.eval()
        if not os.path.exists(img_path):
            raise RuntimeError('----------------------Can\'t find data to predict!----------------------')
        file_list = [img_path]
        if os.path.isdir(img_path):
            file_list = [os.path.join(img_path, x) for x in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, x))]
        for file in file_list:
            img = np.array(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
            # padding
            # img = self.augmentations(image=img)
            img = pad_resize(img, self.cfg.input_size)
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(0)
            # resize to cfg.input_size
            # img = F.interpolate(img.unsqueeze(0), size=self.cfg.input_size, mode="nearest")  # img.size is (1, 3, input_size, input_size)
            detections = self.model(img.cuda())
            detections = non_max_suppression(detections, 0.6, 0.4)
            detections = detections[0]

            # Bounding-box colors
            # cmap = plt.get_cmap("tab20b")
            # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

            img = cv2.imread(file)
            ori_height, ori_width, _ = img.shape
            if detections is not None:
                # print('detections is {}'.format(detections))
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.cfg.input_size, img.shape[:2])
                # unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)
                # bbox_colors = random.sample(colors, n_cls_preds)

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    x1, x2 = np.clip([int(x1), int(x2)], 0, ori_width)
                    y1, y2 = np.clip([int(y1), int(y2)], 0, ori_height)
                    cv2.rectangle(img, (x1, y1), (x2, y2-1), (0, 0, 255), 1)
                    cv2.putText(img, self.classes[int(cls_pred)], (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))
                    cv2.putText(img, '{:.2f}'.format(cls_conf.item()), (x1 + 60, y1-3), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                                (0, 0, 255))

            filename = os.path.basename(file)[:-4]
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            cv2.imwrite(os.path.join(res_dir, f"{filename}.png"), img)
            print('{}{} is predicted!{}'.format('-'*30, file, '-'*30))


def main(model_path, img_path, res_dir):
    test = Test(model_path, cfg)
    test.inference(img_path, res_dir)
    # test.evaluate(iou_thres=0.5, conf_thres=0.5, nms_thres=0.5)


if __name__ == '__main__':
    model_path = r'weights/yolov3_coco_lr1e-4_0.52.pth'
    img_path = r'test_cases'
    res_dir = r'test_cases/output'
    main(model_path, img_path, res_dir)


