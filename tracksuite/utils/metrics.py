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

    left_intersection = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
    top_intersection = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
    right_intersection = np.minimum(bboxes1[:, 2], bboxes2[:, 2])
    bottom_intersection = np.minimum(bboxes1[:, 3], bboxes2[:, 3])

    w_intersection = right_intersection - left_intersection
    h_intersection = bottom_intersection - top_intersection

    # if w_intersection < 0 or h_intersection < 0:
    #     return 0.0

    # intersection_area = w_intersection * h_intersection
    # union_