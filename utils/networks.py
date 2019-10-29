import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
"""initialization need to be improved"""
def weights_init(module):
    class_name = module.__class__.__name__
    if  class_name.find('Linear') != -1:
        #init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
        init.normal_(module.weight.data,0,0.01)
        #if module.bias_term:
        init.constant_(module.bias.data, 0.0)
    elif class_name.find('BatchNorm') != -1:
        init.normal_(module.weight.data, 1.0, 0.02)
        #if module.bias_term:
        init.constant_(module.bias.data, 0.0)
    elif class_name.find('Conv') != -1:
        init.xavier_normal_(module.weight)
        #if module.bias_term:
        init.constant_(module.bias.data, 0.0)

class generator(nn.Module):
    def __init__(self, noise_dim):
        super(generator, self).__init__()
        self.noise_dim = noise_dim
        model = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, 96, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )
        model.apply(weights_init)
        self.model = model

    def forward(self, input):
        return self.model(input.view(-1, self.noise_dim, 1, 1))


class generator_v1(nn.Module):
    def __init__(self, noise_dim):
        super(generator_v1, self).__init__()
        self.noise_dim = noise_dim
        linear_model = nn.Sequential(
            nn.Linear(self.noise_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 49 * 64),
            nn.BatchNorm1d(49 * 64),
            nn.ReLU()
        )
        conv_model = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        linear_model.apply(weights_init)
        conv_model.apply(weights_init)
        self.linear_model = linear_model
        self.conv_model = conv_model
        # self.conv = nn.Conv2d(49*64,64,kernel_size=5,stride=2,padding=2)

    def forward(self, input):
        y = self.linear_model(input)
        y = y.view(-1, 64, 7, 7)
        y = self.conv_model(y)
        return y


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 10, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=7)
        )
        model.apply(weights_init)
        self.model = model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        y = self.model(input)
        y = y.view(-1, 10)
        y = self.softmax(y)
        return y


class discriminator_v1(nn.Module):
    def __init__(self):
        super(discriminator_v1, self).__init__()
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 10, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(490, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 10),
            nn.LeakyReLU(0.2, inplace=True),
        )
        model.apply(weights_init)
        self.model = model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        y = self.model(input)
        y = self.softmax(y)
        return y


