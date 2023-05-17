import os
import cv2
import datetime
import json
from dataset_converters import get_area
import numpy as np
import shutil


def create_image_info(image_id, file_name, image_size):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0]
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }

    return annotation_info


def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]


def convert(via_original_annotations_file, imgdir, annpath):
    """
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    """

    # Get the name(type) and supercategory(super_type) from VIA ANNOTATION
    # You need to modify the attributes depend on your VIA format
    annotations = json.load(open(via_original_annotations_file, encoding="utf-8"))
    annotations = list(annotations.values())  # don't need the dict keys
    annotations = [a for a in annotations if a['regions']]  # filter the empty annotation
    category_names = []
    for a in annotations:
        names = [r['region_attributes']['object'] for r in a['regions']]
        for index, name in enumerate(names):
            if name not in category_names:
                category_names.append(name)

    coco_output = {}

    # get the coco category from dict
    coco_output['categories'] = []
    for i in range(len(category_names)):
        category = {'id': i + 1,
                    'name': category_names[i],
                    }
        coco_output['categories'].append(category)
    coco_output['images'] = []
    coco_output['annotations'] = []
    ##########################################################################################################

    ann = json.load(open(annpath, encoding="utf-8"))
    # annotations id start from zero
    ann_id = 0
    # in VIA annotations, [key]['filename'] are image name
    for img_id, key in enumerate(ann.keys()):
        filename = ann[key]['filename']
        img = cv2.imread(imgdir + filename)
        # make image info and storage it in coco_output['images']
        image_info = create_image_info(img_id, os.path.basename(filename), img.shape[:2])
        coco_output['images'].append(image_info)

        regions = ann[key]["regions"]
        # for one image ,there are many regions,they share the same img id
        for region in regions:
            cate = region['region_attributes']['object']
            # cate must in categories
            assert cate in [i['name'] for i in coco_output['categories']]
            # get the cate_id
            cate_id = 0
            for category in coco_output['categories']:
                if cate == category['name']:
                    cate_id = category['id']
            ####################################################################################################

            iscrowd = 0
            points_x = region['shape_attributes']['all_points_x']
            points_y = region['shape_attributes']['all_points_y']
            area = get_area.GetAreaOfPolyGon(points_x, points_y)

            min_x = min(points_x)
            max_x = max(points_x)
            min_y = min(points_y)
            max_y = max(points_y)
            box = [min_x, min_y, max_x - min_x, max_y - min_y]
            segmentation = get_segmenation(points_x, points_y)

            # make annotations info and storage it in coco_output['annotations']
            ann_info = create_annotation_info(ann_id, img_id, cate_id, iscrowd, area, box, segmentation)
            coco_output['annotations'].append(ann_info)
            ann_id = ann_id + 1

    return coco_output


# automatic split train and val
def train_val_split(annos_name, original_dir, train_dir, val_dir, move):
    annotations = json.load(open(annos_name, encoding="utf-8"))
    annotations = list(annotations.values())

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # get images in annotation
    total_images = [a['filename'] for a in annotations]

    # image index that will move to val
    val_index = np.random.choice(len(annotations), size=len(annotations) // 5, replace=False).tolist()
    train_index = [i for i in range(len(annotations))]
    for i in val_index:
        train_index.remove(i)

    # create train, val annos
    val_annos = {}
    train_annos = {}
    # move images to train, val folder
    if move:
        shutil.rmtree(val_dir)
        os.mkdir(val_dir)
        shutil.rmtree(train_dir)
        os.mkdir(train_dir)
        for i in val_index:
            shutil.copyfile(original_dir + total_images[i],
                            val_dir + total_images[i])
            val_annos[annotations[i]['filename']] = annotations[i]
        for i in train_index:
            shutil.copyfile(original_dir + total_images[i],
                            train_dir + total_images[i])
            train_annos[annotations[i]['filename']] = annotations[i]
    # not move images to train, val folder
    else:
        for i in val_index:
            val_annos[annotations[i]['filename']] = annotations[i]
        for i in train_index:
            train_annos[annotations[i]['filename']] = annotations[i]

    return train_annos, val_annos


if __name__ == '__main__':
    dataset_original_img_path = 'original/'
    dataset_train_img_path = 'train/'
    dataset_val_img_path = 'val/'
    annotations_file = "via_region_data_40.json"

    # get VIA annotations
    VIA_train_annos, VIA_val_annos = train_val_split(annotations_file, dataset_original_img_path,
                                                     dataset_train_img_path, dataset_val_img_path, move=True)

    # save VIA annotations
    with open(dataset_train_img_path + 'VIA_train_annos.json', 'w', encoding="utf-8") as outfile:
        json.dump(VIA_train_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)

    with open(dataset_val_img_path + 'VIA_val_annos.json', 'w', encoding="utf-8") as outfile:
        json.dump(VIA_val_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)

    # convert VIA annotations to COCO annotations
    COCO_train_annos = convert(annotations_file, dataset_train_img_path,
                               dataset_train_img_path + 'VIA_train_annos.json')
    COCO_val_annos = convert(annotations_file, dataset_val_img_path,
                             dataset_val_img_path + 'VIA_val_annos.json')

    # save COCO annotations
    with open(dataset_train_img_path + 'COCO_train_annos.json', 'w', encoding="utf-8") as outfile:
        json.dump(COCO_train_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)

    with open(dataset_val_img_path + 'COCO_val_annos.json', 'w', encoding="utf-8") as outfile:
        json.dump(COCO_val_annos, outfile, sort_keys=True, indent=4, ensure_ascii=False)