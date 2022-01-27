import os
import numpy as np
import json
import random

coco_class_index = []


def convert(size, box):
    '''

    :param size: (width, height)
    :param box: (x, y, w, h)
    :return:
    '''
    dw = 1 / size[0]
    dh = 1 / size[1]
    x = box[0] + box[2] / 2
    y = box[1] + box[3] / 2
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)  # (center_x, center_y, box_width, box_height)


def parse_anns(anns_file, dst_folder):
    '''

    :param anns_file: train.json
    :param dst_file: train.txt
    :return:
    '''
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    with open(anns_file, 'r') as f:
        data = json.load(f)

        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        classes, coco_labels, coco_labels_inverse = {}, {}, {}
        for c in data['categories']:
            coco_labels[len(classes)] = c['id']
            coco_labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)

        for img in data['images']:
            file_name = img['file_name']
            img_width = img['width']
            img_height = img['height']
            img_id = img['id']
            anns_txt_name = file_name.split('.')[0] + '.txt'
            with open(os.path.join(dst_folder, anns_txt_name), 'w') as f:
                for ann in data['annotations']:
                    if ann['image_id'] == img_id and ann['category_id'] == 1:
                        # print('ann[\'category_id\'] is {}'.format(ann['category_id']))
                        # print('type of ann[\'category_id\'] is {}'.format(type(ann['category_id'])))
                        box = convert((img_width, img_height), ann['bbox'])
                        f.write('{} {} {} {} {}\n'.format(coco_labels_inverse[ann['category_id']], box[0], box[1], box[2], box[3]))
            size = os.path.getsize(os.path.join(dst_folder, anns_txt_name))
            if size == 0:
                os.remove(os.path.join(dst_folder, anns_txt_name))


def get_file(data_path, data_file):
    file_list = os.listdir(data_path)
    if 'train' in data_path:
        mode = 'train'
        with open(data_file, 'w') as f:
            for file in file_list:
                file = file.replace('.txt', '.jpg')
                f.write('data/COCO2017/{}/images/{}\n'.format(mode, file))
    elif 'val' in data_path:
        mode = 'val'
        random.shuffle(file_list)
        with open(data_file, 'w') as f:
            for file in file_list[:-1000]:
                file = file.replace('.txt', '.jpg')
                f.write('data/COCO2017/{}/images/{}\n'.format(mode, file))

        with open(data_file.replace('val', 'test'), 'w') as f:
            for file in file_list[-1000:]:
                file = file.replace('.txt', '.jpg')
                f.write('data/COCO2017/{}/images/{}\n'.format(mode, file))
    else:
        raise RuntimeError('wrong mode!')



if __name__ == '__main__':
    # anns_file = '../Dataset/COCO2017/annotations/instances_val2017.json'
    # dst_folder = '../Dataset/COCO2017/val_label'
    # parse_anns(anns_file, dst_folder)

    data_path = '../Dataset/COCO2017/val_label'
    data_file = '../Dataset/COCO2017/val.txt'
    get_file(data_path, data_file)
