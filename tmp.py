import os
import cv2
import numpy as np

def draw_box():
    img = np.ones((760, 1344, 3), dtype=np.uint8) * 124
    height, width, _ = img.shape
    bboxes = np.array([[0.234933, 0.495395, 0.469122, 0.269737],
                       [0.096354, 0.167763, 0.188244, 0.276316],
                       [0.244978, 0.257895, 0.487723, 0.385526],
                       [0.468378, 0.708553, 0.437500, 0.572368]])
    for bbox in bboxes:
        x, y, w, h = bbox * np.array([width, height, width, height])
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    cv2.imshow('test', img)
    cv2.waitKey()


if __name__ == '__main__':
    a = {'a': 'A', 'b': 'B'}
    print(list(a))