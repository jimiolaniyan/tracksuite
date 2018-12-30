import numpy as np
import cv2


def crop_and_scale(prev_img, curr_img, prev_bbox, curr_bbox, size, dim_scale_factor=2):
    left, top, right, bottom = prev_bbox

    width = right - left
    height = bottom - top

    c1 = left + (width / 2)
    c2 = top + (height / 2)

    new_width = dim_scale_factor * width
    new_height = dim_scale_factor * height

    new_left = np.maximum(0, c1 - (new_width / 2))
    new_top = np.maximum(0, c2 - (new_height / 2))
    new_right = np.maximum(0, c1 + (new_width / 2))
    new_bottom = np.maximum(0, c2 + (new_height / 2))

    region = (new_left, new_top, new_right, new_bottom)

    prev = crop_img(prev_img, region=region)
    curr = crop_img(curr_img, region=region)

    prev = scale_img(prev, size)
    curr = scale_img(curr, size)

    w_scale = size[0] / int(new_right - new_left)
    h_scale = size[1] / int(new_bottom - new_top)

    curr_left = np.maximum(0, curr_bbox[0] - new_left) * w_scale
    curr_top = np.maximum(0, curr_bbox[1] - new_top) * h_scale
    curr_right = np.maximum(0, curr_bbox[2] - new_left) * w_scale
    curr_bottom = np.maximum(0, curr_bbox[3] - new_top) * h_scale

    bb = np.array([curr_left, curr_top, curr_right, curr_bottom])
    return prev, curr, bb


def crop_img(img, region):
    """
    Crops an image in the specified image
    :param img: numpy array of the original image to crop
    :param region: tuple specifying left, top, right and bottom coordinates
    :return:
    """
    if region is None:
        return img

    if len(region) != 4:
        raise ValueError('Region must have four elements')

    if not isinstance(img, np.ndarray):
        raise ValueError('Original image must be an ndarray')

    left, top, right, bottom = region

    if left >= right:
        raise ValueError('Left pixel: {} cannot be greater than right pixel: {}'.format(left, right))

    if top >= bottom:
        raise ValueError('Top pixel: {} cannot be greater than bottom pixel: {}'.format(top, bottom))

    return img[int(top): int(bottom), int(left): int(right)]


def scale_img(img, size):
    if isinstance(size, tuple):
        try:
            n_img = cv2.resize(img, dsize=size)
            return n_img
        except:
            raise Exception('failed to resize image with size {}'.format(img.shape))
    elif isinstance(size, int):
        return cv2.resize(img, dsize=(size, size))
