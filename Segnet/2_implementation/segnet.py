import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd, optim
from modules import *


class segnet(nn.Module):

    def __init__(self, in_channels, n_classes):
        super(segnet, self).__init__()

        self.layer_1 = SegNetBlockDown(in_channels, 64, 2)
        self.layer_2 = SegNetBlockDown(64, 128, 2)
        self.layer_3 = SegNetBlockDown(128, 256, 3)
        self.layer_4 = SegNetBlockDown(256, 512, 3)
        self.layer_5 = SegNetBlockDown(512, 512, 3)

        self.layer_6 = SegNetBlockUp(512, 512, 3)
        self.layer_7 = SegNetBlockUp(512, 256, 3)
        self.layer_8 = SegNetBlockUp(256, 128, 3)
        self.layer_9 = SegNetBlockUp(128, 64, 2)
        self.layer_10 = SegNetBlockUp(64, n_classes, 2)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.layer_1(inputs=inputs,
                                                       layer_size=2)
        down2, indices_2, unpool_shape2 = self.layer_2(inputs=down1,
                                                       layer_size=2)
        down3, indices_3, unpool_shape3 = self.layer_3(inputs=down2,
                                                       layer_size=3)
        down4, indices_4, unpool_shape4 = self.layer_4(inputs=down3,
                                                       layer_size=3)
        down5, indices_5, unpool_shape5 = self.layer_5(inputs=down4,
                                                       layer_size=3)

        up5 = self.layer_6(inputs=down5, indices=indices_5,
                           output_shape=unpool_shape5, layer_size=3)
        up4 = self.layer_7(inputs=up5, indices=indices_4,
                           output_shape=unpool_shape4, layer_size=3)
        up3 = self.layer_8(inputs=up4, indices=indices_3,
                           output_shape=unpool_shape3, layer_size=3)
        up2 = self.layer_9(inputs=up3, indices=indices_2,
                           output_shape=unpool_shape2, layer_size=2)
        up1 = self.layer_10(inputs=up2, indices=indices_1,
                            output_shape=unpool_shape1, layer_size=2)

        return up1
