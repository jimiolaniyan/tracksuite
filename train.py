import random
import numpy as np
import torch
# from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

from tracksuite.datasets.alov import ALOVDataSet
import cv2
epochs = 1


def train_model():
    # setup_seeds(747)

    device = get_device()

    # TODO: replace with either args or config file
    videos = '~/Documents/Research/DataSets/ALOV/imagedata++'
    annotations = '~/Documents/Research/DataSets/ALOV/alov300++_rectangleAnnotation_full'
    alov_dataset = ALOVDataSet(videos_root=videos, annotations_root=annotations)

    alov_loader = DataLoader(alov_dataset, batch_size=32, num_workers=4, sampler = RandomSampler(alov_dataset))
    i = 19
    j = i + 1
    for epoch in range(epochs):
        for data in alov_loader:
            prev_img_batch, curr_img_batch = data[0]
            labels = data[1]
            img, img2 = prev_img_batch[i].numpy(), curr_img_batch[i].numpy()
            img_, img2_ = prev_img_batch[j].numpy(), curr_img_batch[j].numpy()

            cv2.rectangle(img2, (int(labels[i][0]), int(labels[i][1])), (int(labels[i][2]), int(labels[i][3])), (89, 122, 41), 1)
            cv2.rectangle(img2_, (int(labels[j][0]), int(labels[j][1])), (int(labels[j][2]), int(labels[j][3])), (89, 122, 41), 1)
            cv2.imshow('im1', img)
            cv2.imshow('im2', img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('im1', img_)
            cv2.imshow('im2', img2_)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            break
        break


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    train_model()
