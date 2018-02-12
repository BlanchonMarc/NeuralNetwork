import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd, optim

from modules import *


class segnet(nn.Module):

    def __init__(self, in_channels, n_classes):
        super(segnet, self).__init__()

        self.layer_1 = segnetDown2(in_channels, 64)
        self.layer_2 = segnetDown2(64, 128)
        self.layer_3 = segnetDown3(128, 256)
        self.layer_4 = segnetDown3(256, 512)
        self.layer_5 = segnetDown3(512, 512)

        self.layer_6 = segnetUp3(512, 512)
        self.layer_7 = segnetUp3(512, 256)
        self.layer_8 = segnetUp3(256, 128)
        self.layer_9 = segnetUp2(128, 64)
        self.layer_10 = segnetUp2(64, n_classes)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.layer_1(inputs)
        down2, indices_2, unpool_shape2 = self.layer_2(down1)
        down3, indices_3, unpool_shape3 = self.layer_3(down2)
        down4, indices_4, unpool_shape4 = self.layer_4(down3)
        down5, indices_5, unpool_shape5 = self.layer_5(down4)

        up5 = self.layer_6(down5, indices_5, unpool_shape5)
        up4 = self.layer_7(up5, indices_4, unpool_shape4)
        up3 = self.layer_8(up4, indices_3, unpool_shape3)
        up2 = self.layer_9(up3, indices_2, unpool_shape2)
        up1 = self.layer_10(up2, indices_1, unpool_shape1)

        return up1


# Execution
batch_size = 1
input_size = 8
num_classes = 8
learning_rate = 0.0001
nb = 64

input = autograd.Variable(torch.rand(batch_size, input_size, nb, nb))
target = autograd.Variable(torch.rand(batch_size, num_classes, nb, nb)).long()


model = segnet(in_channels=input_size, n_classes=num_classes)

opt = optim.Adam(params=model.parameters(), lr=learning_rate)


for epoch in range(100):
    out = model(input)

    loss = F.cross_entropy(out, target[:, 0])

    print ('Loss : ' + str(loss.data))

    model.zero_grad()
    loss.backward()

    opt.step()
