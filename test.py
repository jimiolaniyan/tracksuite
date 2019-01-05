import os
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from tracksuite.trackers.goturn import Goturn
from tracksuite.utils.image import crop_and_scale, net_bb_to_orig_bb


def get_init_bb(gt):
    # OTB gt is [left, top, width, height]
    bb = gt[0]

    # convert to [left, top, right, bottom]
    bb = np.array([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]])

    return bb


def test(img_path, gt_path, model_path):
    gt = np.loadtxt(gt_path)

    images = os.listdir(img_path)
    images.sort()

    bboxes = np.zeros((len(images), 4))
    bb = None

    model = Goturn()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.39073333, 0.40982353, 0.43166235],
                             std=[0.20822039, 0.21170275, 0.21818118])
    ])

    for i in range(len(images) - 1):
        if i == 0:
            bb = get_init_bb(gt)
            bboxes[0, :] = bb

        prev_img = cv2.imread(os.path.join(img_path, images[i]))
        curr_img = cv2.imread(os.path.join(img_path, images[i + 1]))

        exemplar, search = crop_and_scale(prev_img, curr_img, bb, None, (227, 227))

        exemplar = transform(exemplar)
        search = transform(search)

        exemplar = exemplar[None, :, :, :]
        search = search[None, :, :, :]

        new_bb = model(exemplar, search)
        new_bb = new_bb.data.cpu().numpy()[0]
        new_bb = new_bb * (227 / 10)

        bb = net_bb_to_orig_bb(new_bb, bb)
        bboxes[i, :] = bb


if __name__ == "__main__":
    img_path = '/home/jimiolaniyan/Downloads/Dog/img'
    gt_path = '/home/jimiolaniyan/Downloads/Dog/groundtruth_rect.txt'
    model_path = '/content/drive/My Drive'
    test(img_path=img_path, gt_path=gt_path, model_path=None)
