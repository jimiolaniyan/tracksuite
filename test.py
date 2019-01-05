import os
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from tracksuite.trackers.goturn import Goturn
from tracksuite.utils.image import crop_and_scale, net_bb_to_orig_bb


def get_init_bb(gt, ind):

    # OTB gt is [left, top, width, height]
    bb = gt[ind]

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
    # model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.39073333, 0.40982353, 0.43166235],
                                     std=[0.20822039, 0.21170275, 0.21818118])
             ])

    # len(images) - 1
    for i in range(10):
        # if i == 0:
        bb = get_init_bb(gt, i)
        # bboxes[0, :] = bb

        prev_img = cv2.imread(os.path.join(img_path, images[i]))
        curr_img = cv2.imread(os.path.join(img_path, images[i + 1]))

        bb1 = gt[i+1]
        bb1 = np.array([bb1[0], bb1[1], bb1[0] + bb1[2], bb1[1] + bb1[3]])

        exemplar, search, currbb = crop_and_scale(prev_img, curr_img, bb, bb1, (227, 227))
        print(currbb)
        print(bb1)
        # cv2.rectangle(search, (int(currbb[0]), int(currbb[1])), (int(currbb[2]), int(currbb[3])), (23,34,103))
        # # cv2.rectangle(prev_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (23,34,103))
        # cv2.imshow('prev_img', search)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        currbb = currbb * (10/227)


        # exemplar = transform(exemplar)
        # search = transform(search)
        #
        # exemplar = exemplar[None, :, :, :]
        # search = search[None, :, :, :]
        #
        # new_bb = model(exemplar, search)
        # new_bb = new_bb.data.cpu().numpy()[0]
        currbb = currbb * (227/10)


        curr_bbox = net_bb_to_orig_bb(currbb, bb)
        print(curr_bbox)
        cv2.rectangle(curr_img, (int(curr_bbox[0]), int(curr_bbox[1])), (int(curr_bbox[2]), int(curr_bbox[3])), (23,34,103))
        # cv2.rectangle(prev_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (23,34,103))
        cv2.imshow('prev_img', curr_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('End of loop', '*' * 30)
        # currbb = curr_bbox

        # print(bb)
        # print(bb1)
        # print(curr_bbox)
        # print('End of loop', '*' * 30)
        # # bb = curr_bbox

if __name__ == "__main__":
    img_path = '/home/jimiolaniyan/Downloads/Dog/img'
    gt_path = '/home/jimiolaniyan/Downloads/Dog/groundtruth_rect.txt'
    test(img_path=img_path, gt_path=gt_path, model_path=None)

