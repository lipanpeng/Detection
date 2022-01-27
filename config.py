#coding:utf-8
"""
@ Author: Peyton
"""

input_size = 416
input_channel = 3
num_classes = 1

batch_size = 32
begin_epoch = 0
epoch = 500

# Training COCO2017 with lr 1e-2, loss will not diverge
lr = 1e-5
weight_decay = 0
momentum = 0.9
nesterov = False
dropout = False

dataset = 'coco'
classes_path = 'dataset/coco.names'
train_file = 'data/COCO2017/train.txt'
val_file = 'data/COCO2017/val.txt'
test_file = 'data/COCO2017/test.txt'
name = 'yolov3'
model_config = 'models/yolov3/yolov3.cfg'

pretrained_model = 'weights/yolo_weights'
model_path = 'weights/yolov3_coco_lr1e-5.pth'

cuda = True
workers = 2
gpu_ids = '0'
