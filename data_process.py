# -*- coding:utf-8 -*-
import os
import numpy as np
import json
import cv2
import ipdb
import random
import time
import shutil
import datetime


IMG_EXTENSION = ('.jpg', '.png', '.jpeg', '.JPG')

def unify_ann(ori_ann, dst_ann, start_idx):
    with open(ori_ann, 'r', encoding='utf-8') as f:
        anns_data = json.load(f)
        print(len(anns_data))
        print(type(anns_data))
    with open(dst_ann, 'r') as f:
        ann_dict = json.load(f)
        for name, ann in anns_data.items():
            img_name = ann['filename']
            img_path = os.path.join(os.path.split(ori_ann)[0], img_name)
            # img = cv2.imread(img_path)
            try:
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
                name = '{}'.format(start_idx)
                ann['filename'] = '{}{}'.format(name, os.path.splitext(img_name)[-1])
                ann_dict.update({name: ann})
                cv2.imwrite(os.path.join(os.path.split(dst_ann)[0], ann['filename']), img)
                start_idx += 1
            except FileNotFoundError:
                print('{} is not founded'.format(img_name))
    with open(dst_ann, 'w') as f:
        json.dump(ann_dict, f)


def check_ann(ann_file, invalid_file):
    if invalid_file is None:
        invalid_list = []
    else:
        with open(invalid_file, 'r') as f:
            invalid_list = [i.strip() for i in f.readlines()]
    with open(ann_file, 'r') as f:
        ann_data = json.load(f)
        for name, ann in ann_data.items():
            img_name = ann['filename']
            # ipdb.set_trace()
            if img_name in invalid_list:
                img = cv2.imread(os.path.join(os.path.split(ann_file)[0], img_name))
                print(img.shape)
                if len(ann['regions']) == 0:
                    invalid_list.append(img_name)
                for region in ann['regions']:
                    x = region['shape_attributes']['x']
                    y = region['shape_attributes']['y']
                    w = region['shape_attributes']['width']
                    h = region['shape_attributes']['height']
                    box = [x, y, x+w, y+h]
                    box_color = (255, 0, 255)
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=box_color, thickness=2)
                cv2.imshow('test'.format(img_name), img)
                cv2.waitKey()


def arrange_data(data_dir, dist_dir):
    for folder in os.listdir(data_dir):
        print('--------------{}--------------'.format(folder))
        for file_name in os.listdir(os.path.join(data_dir, folder)):
            if file_name.endswith(IMG_EXTENSION):
                shutil.copy(os.path.join(data_dir, folder, file_name), os.path.join(dist_dir, 'images'))

                shutil.copy(os.path.join(data_dir, folder, os.path.splitext(file_name)[0]+'.json'), os.path.join(dist_dir, 'labels'))


def get_frame_from_video(video_dir, dst_dir):
    cap = cv2.VideoCapture(video_dir)
    interval = 15
    frame = 0
    while True:
        ret_val, img = cap.read()
        frame += 1
        if frame % interval:
            continue
        if not ret_val:
            break
        file_name = os.path.join(dst_dir, '{}.jpg'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')))
        cv2.imwrite(file_name, img)
        print('--------------save file {}--------------'.format(os.path.basename(file_name)))



if __name__ == '__main__':

    if True:
        video_dir = r'../Dataset/Detection/sales/6#销售区地磅上方_124398_img/6#销售区地磅上方_124398.mp4'
        dst_dir = r'../Dataset/Detection/sales/video_img'
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        get_frame_from_video(video_dir, dst_dir)

    ## Collate data
    if False:
        ori_ann = r'../Dataset/detection/0206_病死猪照片/via_project_13Feb2023_8h21m_json.json'
        dst_ann = r'../Dataset/detection/carcass/dataset/anns.json'
        exist_dataset = [i for i in os.listdir(os.path.split(dst_ann)[0]) if i.endswith(IMG_EXTENSION)]
        start_idx = len(exist_dataset)
        unify_ann(ori_ann, dst_ann, start_idx)

    ## check data
    if False:
        ann_file = r'../Dataset/detection/carcass/dataset/anns.json'
        invalid_file = r'../Dataset/detection/carcass/dataset/invalid.txt'
        check_ann(ann_file, invalid_file)

    ## split dataset
    if False:
        data_dir = r'../Dataset/detection/carcass/dataset'
        train_txt = r'../Dataset/detection/carcass/dataset/train.txt'
        val_txt = r'../Dataset/detection/carcass/dataset/val.txt'
        test_txt = r'../Dataset/detection/carcass/dataset/test.txt'

        train_r = 0.8
        val_r = 0.1
        train_list = []
        val_list = []
        test_list = []

        dir_list = [i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))]
        for dir in dir_list:
            data_list = os.listdir(os.path.join(data_dir, dir))
            random.shuffle(data_list)
            train_list.extend([i for i in data_list[:int(train_r*len(data_list))]])
            val_list.extend([i for i in data_list[int(train_r*len(data_list)):int((train_r+val_r)*len(data_list))]])
            test_list.extend([i for i in data_list[int((train_r+val_r)*len(data_list)):]])

        with open(train_txt, 'w') as f:
            for i in train_list:
                f.write(i+'\n')
        with open(val_txt, 'w') as f:
            for i in val_list:
                f.write(i+'\n')
        with open(test_txt, 'w') as f:
            for i in test_list:
                f.write(i+'\n')

    ## sort data
    if False:
        test_file = r'test.txt'
        ann_file = r'anns.json'
        with open(test_file, 'r') as f:
            img_list = f.readlines()
            # ipdb.set_trace()
        # img_list = [i for i in os.listdir(data_dir) if i.endswith(IMG_EXTENSION)]
        max_num = 0
        test_list = [[] for i in range(5)]
        num_thresh = [10, 30, 60, 90]
        with open(ann_file, 'r') as f:
            anns = json.load(f)
            for img_name in img_list:
                img_idx = img_name.split('.')[0]
                obj_num = len(anns[img_idx]['regions'])
                if obj_num < num_thresh[0]:
                    test_list[0].append(img_name)
                elif obj_num < num_thresh[1]:
                    test_list[1].append(img_name)
                elif obj_num < num_thresh[2]:
                    test_list[2].append(img_name)
                elif obj_num < num_thresh[3]:
                    test_list[3].append(img_name)
                else:
                    test_list[4].append(img_name)
        for i in range(len(test_list)):
            with open('test_{}.txt'.format(i), 'w') as f:
                for img_name in test_list[i]:
                    f.write(img_name)


