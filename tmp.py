import cv2




if __name__ == '__main__':
    data_dir = r'../Dataset/COCO2017/train2017/000000000036.jpg'
    ori = [0.6712785862785863, 0.6179453125000001, 0.6457588357588359, 0.7268593750000001]
    img = cv2.imread(data_dir)
    img_h, img_w, _ = img.shape
    x_center, y_center, w, h = ori
    x_center, y_center = x_center * img_w, y_center * img_h
    w, h = w * img_w, h * img_h
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imshow('test', img)
    cv2.waitKey()
