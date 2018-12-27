import torch
import torch.nn as nn
from torchvision.models import alexnet


class Goturn(nn.Module):

    def __init__(self):
        super(Goturn, self).__init__()

        # net here (alexnet) is referred to as caffenet in the paper
        net = alexnet(pretrained=True)

        # these are the convolutional layers of alexnet
        self.conv = net.features
        self.fc = nn.Sequential(
            nn.Linear(256*6*6*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4)
        )

    def forward(self, x, y):
        x = x.permute(0, 3, 2, 1)
        y = y.permute(0, 3, 2, 1)

        x_ = self.conv(x).view(-1, 256*6*6)
        y_ = self.conv(y).view(-1, 256*6*6)

        z = torch.cat((x_, y_), 1)
        z = self.fc(z)
        return z
