import numpy as np


def calculate_iou(bboxes1, bboxes2):
    """
    This calculates the intersection over union of N bounding boxes
    in the form N x [left, top, right, bottom],  e.g for N=2:
    >> bb = [[21,34,45,67], [67,120, 89, 190]]
    :param bboxes1: np array: N x 4 ground truth bounding boxes
    :param bboxes2: np array: N x 4 target bounding boxes
    :return: iou: ratio between 0 and 1
    """

    if len(bboxes1.shape) == 1:
        bboxes1 = bboxes1.reshape(1, bboxes1.shape[0])

    if len(bboxes2.shape) == 1:
        bboxes2 = bboxes2.reshape(1, bboxes2.shape[0])

    if bboxes1.shape[0] != bboxes2.shape[0] or bboxes1.shape[1] != bboxes2.shape[1]:
        raise ValueError('Bounding boxes must be of equal dimension')

    left_intersection = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    top_intersection = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    right_intersection = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    bottom_intersection = np.minimum(bboxes1[:, 3], bboxes2[:, 3])

    w_intersection = right_intersection - left_intersection
    h_intersection = bottom_intersection - top_intersection

    intersection_area = w_intersection * h_intersection

    bboxes1_area = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    bboxes2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    union_area = bboxes1_area + bboxes2_area - intersection_area

    iou = np.clip(intersection_area/union_area, 0, 1)
    return iou