import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler 

from tracksuite.datasets.alov import ALOVDataSet

epochs = 1
def train_model():
    setup_seeds(747)
    
    device = get_device()

    # TODO: replace with either args or config file
    videos = '~/Documents/Research/DataSets/ALOV/imagedata++'
    annotations = '~/Documents/Research/DataSets/ALOV/alov300++_rectangleAnnotation_full'
    alovDataset = ALOVDataSet(videos_root=videos, annotations_root=annotations)

    alovLoader = DataLoader(alovDataset, batch_size=10, num_workers=4)
    for epoch in range(epochs):
        for data in  alovLoader:
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