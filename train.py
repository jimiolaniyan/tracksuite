import os
import sys
import time
import datetime
import copy
import random
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, random_split

# from tensorboardX import SummaryWriter

from tracksuite.datasets.alov import ALOVDataSet
from tracksuite.trackers.goturn import Goturn
from tracksuite.utils.metrics import calculate_iou

epochs = 50


def train_model():
    setup_seeds(747)

    device = get_device()

    # TODO: replace with either args or config file
    # videos = '/content/drive/My Drive/Research/Datasets/ALOV/imagedata++'
    # annotations = '/content/drive/My Drive/Research/Datasets/ALOV/alov300++_rectangleAnnotation_full'
    videos = '~/Documents/Research/DataSets/ALOV/imagedata++'
    annotations = '~/Documents/Research/DataSets/ALOV/alov300++_rectangleAnnotation_full'

    alov_dataset = ALOVDataSet(videos_root=videos, annotations_root=annotations)
    train_size = int(0.8 * len(alov_dataset))
    val_size = len(alov_dataset) - train_size
    train_dataset, val_dataset = random_split(alov_dataset, [train_size, val_size])

    print(len(alov_dataset))
    loader = DataLoader(alov_dataset, batch_size=50, num_workers=6, shuffle=False)

    print(len(loader))
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        prev_imgs, _ = data[0]
        prev_imgs = prev_imgs.to(dtype=torch.float32)
        batch_samples = prev_imgs.size(0)
        nb_samples += batch_samples

        prev_imgs = prev_imgs.view(-1, prev_imgs.size(3))
        mean += prev_imgs.mean(0)
        std += prev_imgs.std(0)

    mean /= len(loader)
    std /= len(loader)
    print(mean, std)

    # print(nb_samples, mean, std)
    # train_loader = DataLoader(train_dataset, batch_size=50, num_workers=6, sampler=RandomSampler(train_dataset),
    #                           drop_last=False)
    #
    # val_loader = DataLoader(val_dataset, batch_size=50, num_workers=6, sampler=RandomSampler(val_dataset),
    #                         drop_last=False)
    #
    # model = Goturn()
    #
    # for param in model.conv.parameters():
    #     param.requires_grad = False
    #
    # model = model.to(device)
    #
    # criterion = nn.L1Loss()
    # optimizer = optim.SGD(model.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #
    # saved_model = '/content/drive/My Drive/colab/goturn_2018_12_29_14_36_epoch_10.pth'
    # if os.path.isfile(saved_model):
    #     logging.info('Loading checkpoint {}'.format(saved_model))
    #     checkpoint = torch.load(saved_model)
    #     model.load_state_dict(checkpoint['model_state'])
    #     # optimizer.load_state_dict(checkpoint['optimizer_state'])
    #     logging.info('Loaded model from {}'.format(saved_model))
    #
    # best_acc = -100.0
    # since = time.time()
    # for epoch in range(epochs):
    #     epoch_start = time.time()
    #     logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
    #     logging.info('-' * 20)
    #
    #     for phase in ['train', 'val']:
    #         print(phase)
    #         if phase == 'train':
    #             dataset = train_dataset
    #             data_loader = train_loader
    #             scheduler.step()
    #             model.train()
    #         else:
    #             dataset = val_dataset
    #             data_loader = val_loader
    #             model.eval()
    #
    #         running_loss = 0.0
    #         running_acc = 0.0
    #
    #         for i, data in enumerate(data_loader):
    #             prev_imgs, curr_imgs = data[0]
    #             labels = data[1]
    #
    #             prev_imgs = prev_imgs.to(device, dtype=torch.float32)
    #             curr_imgs = curr_imgs.to(device, dtype=torch.float32)
    #
    #             labels = labels.to(device, dtype=torch.float32)
    #
    #             optimizer.zero_grad()
    #
    #             with torch.set_grad_enabled(phase == 'train'):
    #                 outputs = model(prev_imgs, curr_imgs)
    #                 if phase == 'train':
    #                     outputs = outputs * 10
    #
    #                 loss = criterion(outputs, labels)
    #
    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()
    #
    #             running_loss += loss.item() * labels.size(0)
    #             # print(outputs.data)
    #
    #             if phase == 'val':
    #                 iou = calculate_iou(outputs, labels)
    #                 print(iou)
    #                 acc = torch.mean(iou)
    #                 print(acc)
    #                 # running_acc += acc
    #             break
    #             # writer.add_scalar('loss/running_loss', loss.item() * labels.size(0), i + 1)
    #             # writer.add_scalar('acc/running_acc', acc.item() * labels.size(0), i + 1)
    #
    #         epoch_time = time.time() - epoch_start
    #         logging.info('Epoch Time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
    #
    #         epoch_loss = running_loss / len(dataset)
    #         epoch_acc = running_acc / len(dataset)
    #         # writer.add_scalar('loss/epoch_loss', epoch_loss, epoch + 1)
    #         # writer.add_scalar('acc/epoch_acc', epoch_acc, epoch + 1)
    #         logging.debug('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    #
    #     break
    #     if phase == 'val' and epoch_acc > best_acc:
    #         model_wts = copy.deepcopy(model.state_dict())
    #         now = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
    #         state = {
    #             "epoch": epoch + 1,
    #             "model_state": model_wts,
    #             "optimizer_state": optimizer.state_dict(),
    #             "loss": epoch_loss
    #         }
    #         torch.save(state, '/content/drive/My Drive/colab/goturn_{}_epoch_{}.pth'.format(now, epoch + 1))
    #
    # time_taken = time.time() - since
    #
    # logging.info('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_taken // 60, time_taken % 60))
    # logging.info('Best validation accuracy: {}'.format(best_acc))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    LOG_DIR = '/content/drive/My Drive/colab/log'
    # writer = SummaryWriter(log_dir=LOG_DIR)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG,
                        stream=sys.stdout,
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('Call main loop')
    train_model()
