import torch
import numpy as np
from utils.utils import bbox_wh_iou, bbox_iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    # shape of pred_boxes is (batch_size, num_anchors, h, w, 4)
    # shape of pred_cls is (batch_size, num_anchors, h, w, num_classes)
    # shape of target is (batch_size*num_gt_box, 6), 6 for (batch_idx, label, x, y, w, h)
    # shape of anchor is (3, 2)
    # ignore_thres is a float

    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # batch size
    nA = pred_boxes.size(1)  # num_anchors
    nC = pred_cls.size(-1)  # num_classes
    nG = pred_boxes.size(2)  # grid size

    # Output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)  # (batch_size, num_anchors, grid_size, grid_size)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)  # (batch_size, num_anchors, grid_size, grid_size, num_classes)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG  # shape is (batch_size*num_gt_boxes, 4)
    gxy = target_boxes[:, :2]  # shape is (batch_size*num_gt_boxes, 2)
    gwh = target_boxes[:, 2:]  # shape is (batch_size*num_gt_boxes, 2)
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # shape is (3, batch_size*num_gt_boxes)
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf