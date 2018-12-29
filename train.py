import sys
import time
import random
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from tracksuite.datasets.alov import ALOVDataSet
from tracksuite.trackers.goturn import Goturn

epochs = 3


def train_model():
    # setup_seeds(747)

    device = get_device()

    # TODO: replace with either args or config file
    videos = '~/Documents/Research/DataSets/ALOV/imagedata++'
    annotations = '~/Documents/Research/DataSets/ALOV/alov300++_rectangleAnnotation_full'
    alov_dataset = ALOVDataSet(videos_root=videos, annotations_root=annotations)

    alov_loader = DataLoader(alov_dataset, batch_size=32, num_workers=6, sampler=RandomSampler(alov_dataset),
                             drop_last=False)

    model = Goturn()

    for param in model.conv.parameters():
        param.requires_grad = False

    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0005)

    since = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
        logging.info('-' * 20)

        model.train()

        running_loss = 0.0

        for data in tqdm(alov_loader):
            prev_imgs, curr_imgs = data[0]
            labels = data[1]
            optimizer.zero_grad()

            prev_imgs = prev_imgs.to(device, dtype=torch.float32)
            curr_imgs = curr_imgs.to(device, dtype=torch.float32)

            labels = labels.to(device, dtype=torch.float32)

            outputs = model(prev_imgs, curr_imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(alov_dataset)

        logging.debug('Loss: {:.4f}'.format(epoch_loss))
        logging.info('Epoch Time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))

    time_taken = time.time() - since

    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_taken // 60, time_taken % 60))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout,
                    datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('Training started')
    train_model()
