import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc_1 = nn.Linear(74, 1024)
        self.fc_2 = nn.Linear(1024, 7*7*128)

        self.bn_1 = nn.BatchNorm1d(1024)
        self.bn_2 = nn.BatchNorm2d(128)
        self.bn_3 = nn.BatchNorm2d(64)

        self.upconv_1 = nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1, bias=False)
        self.upconv_2 = nn.ConvTranspose2d(64, 1, (4,4), stride=2, padding=1, bias=False)

    def forward(self,x):
        # Construct network described in paper
        x = self.bn_1(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.view(-1,128,7,7)
        x = self.bn_2(F.relu(x))
        x = self.bn_3(F.relu(self.upconv_1(x)))
        x = F.sigmoid(self.upconv_2(x))
        return x

class DiscriminatorFrontEnd(nn.Module):
    def __init__(self):
        super(DiscriminatorFrontEnd, self).__init__()

        self.fc = nn.Linear(7*7*128, 1024)

        self.bn_1 = nn.BatchNorm2d(128)
        self.bn_2 = nn.BatchNorm1d(1024)

        self.conv_1 = nn.Conv2d(1, 64, (4,4), stride=2, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(64, 128, (4,4), stride=2, padding=1, bias=False)

    def forward(self,x):
        # Construct network described in paper
        # Input = 1x28x28
        x = F.leaky_relu(self.conv_1(x))
        # x.shape = 64x14x14
        x = F.leaky_relu(self.conv_2(x))
        # x.shape = 128x7x7
        x = self.bn_1(x)
        x = x.view(-1,7*7*128)
        x = self.fc(x)
        x = self.bn_2(x)
        return x

class DiscriminatorBackend(nn.Module):
    def __init__(self):
        super(DiscriminatorBackend, self).__init__()
        self.fc = nn.Linear(1024, 1)
    def forward(self,x):
        x = F.sigmoid(self.fc(x))
        return x

class DiscriminatorInfo(nn.Module):
    def __init__(self):
        super(DiscriminatorInfo, self).__init__()
        self.fc_1 = nn.Linear(1024, 128)
        self.fc_2 = nn.Linear(128,12)

        self.bn = nn.BatchNorm1d(128)

    def forward(self,x):
        x = self.fc_1(x)
        x = F.leaky_relu(self.bn(x))
        x = self.fc_2(x)
        return x

        
