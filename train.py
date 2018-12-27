import random
import numpy as np
import torch
# from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

from timeit import default_timer as timer

from tracksuite.datasets.alov import ALOVDataSet

epochs = 1


def train_model():
    setup_seeds(747)

    device = get_device()

    # TODO: replace with either args or config file
    videos = '~/Documents/Research/DataSets/ALOV/imagedata++'
    annotations = '~/Documents/Research/DataSets/ALOV/alov300++_rectangleAnnotation_full'
    alov_dataset = ALOVDataSet(videos_root=videos, annotations_root=annotations)

    alov_loader = DataLoader(alov_dataset, batch_size=32, num_workers=1)
    for epoch in range(epochs):
        for data in alov_loader:
            pass


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    train_model()
