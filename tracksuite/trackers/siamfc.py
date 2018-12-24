import torch
import torch.nn as nn
import numpy as np
import os
from scipy.io import loadmat

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2,),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )

    def forward(self):
        pass


def load_net(path):
    mat = loadmat(path)
    net = mat['net']
    params = net['params'][0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]
    return params_names_list, params_values_list


def load_video(path):
    ground_truth = np.loadtxt('{}/groundtruth_rect.txt'.format(path))
    img_path = '{}/img/'.format(path)
    img_file_paths = ['{}{}'.format(img_path, f) for f in  os.listdir('{}'.format(img_path))]
    img_file_paths.sort()
    return img_file_paths

if __name__ == "__main__":
    load_video('./Dog')
    load_net('./gray025_net')