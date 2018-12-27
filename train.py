import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

from tracksuite.datasets.alov import ALOVDataSet
from tracksuite.trackers.goturn import Goturn

epochs = 1


def train_model():
    # setup_seeds(747)

    device = get_device()

    # TODO: replace with either args or config file
    videos = '~/Documents/Research/DataSets/ALOV/imagedata++'
    annotations = '~/Documents/Research/DataSets/ALOV/alov300++_rectangleAnnotation_full'
    alov_dataset = ALOVDataSet(videos_root=videos, annotations_root=annotations)

    alov_loader = DataLoader(alov_dataset, batch_size=32, num_workers=6, sampler=RandomSampler(alov_dataset))

    model = Goturn().to(device)

    for epoch in range(epochs):
        for data in tqdm(alov_loader):
            prev_imgs, curr_imgs = data[0]
            labels = data[1]
            model.train()

            prev_imgs = prev_imgs.to(device, dtype=torch.float32)
            curr_imgs = curr_imgs.to(device, dtype=torch.float32)

            outputs = model(prev_imgs, curr_imgs)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    train_model()
