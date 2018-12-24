import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from tracksuite.datasets.alov import ALOVDataSet
import cv2
from matplotlib import pyplot as plt

epochs = 1


def train_model():
    # setup_seeds(747)
    #
    # device = get_device()
    #
    # # TODO: replace with either args or config file
    videos = '~/Documents/Research/DataSets/ALOV/imagedata++'
    annotations = '~/Documents/Research/DataSets/ALOV/alov300++_rectangleAnnotation_full'
    alovDataset = ALOVDataSet(videos_root=videos, annotations_root=annotations)

    # img = cv2.imread(
    #     '/home/jimiolaniyan/Documents/Research/DataSets/ALOV/imagedata++/01-Light/01-Light_video00001/00000001.jpg')
    # img2 = cv2.imread(
    #     '/home/jimiolaniyan/Documents/Research/DataSets/ALOV/imagedata++/01-Light/01-Light_video00001/00000006.jpg')
    # cv2.rectangle(img, (int(1012.9), int(187.84)), (int(1080.2), int(269.56)), (20, 255, 44), 1)
    #
    # w = 1080 - 1013
    # h = 270 - 188
    #
    # c1 = int(188 + (w / 2))
    # c2 = int(1013 + (h / 2))
    #
    # k1 = 2
    #
    # kw = k1 * w
    # kh = k1 * h
    #
    # a = c2 - (kw // 2)
    # b = c1 - (kh // 2)
    # c = c2 + (kw // 2)
    # d = c1 + (kh // 2)
    #
    # bb = np.array([1107, 175, 1039, 257.15])
    # # cv2.rectangle(img, (a, b), (c,d), (20, 255, 44), 1)
    # # cv2.rectangle(img2, (a, b), (c,d), (20, 255, 44), 1)
    #
    # # print(c1, c2, w, h, a, b, c, d)
    #
    # # cv2.circle(img, (c2,c1), 2, (44,255,20), 3)
    #
    # cropped = img2[b:d, a:c]
    #
    # left = int(bb[0] - a)
    # top = int(bb[1] - b)
    # right = int(bb[2] - a)
    # bottom = int(bb[3] - b)
    #
    # print(left, right, top, bottom)
    #
    # size = (227, 227)
    # s_cropped = cv2.resize(cropped, size)
    #
    # w_scale = size[0]/ int(c-a)
    # h_scale = size[1]/ int(d-b)
    #
    # # cv2.imshow('image', cropped)
    #
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    #
    # # cv2.rectangle(img2, (1107, 175), (1040, 257), (20, 25, 44), 1)
    # cv2.rectangle(s_cropped, (int(left * w_scale), int(top * h_scale)), (int(right * w_scale), int(bottom * h_scale)), (20, 25, 44), 1)
    # cv2.imshow('image', s_cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # bb = np.array([1080.2, 187.84, 1012.9, 187.84, 1012.9, 269.56, 1080.2, 269.56])
    #
    # alovLoader = DataLoader(alovDataset, batch_size=50, num_workers=4)
    # for epoch in range(epochs):
    #     for data in alovLoader:
    #         pass


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    train_model()
