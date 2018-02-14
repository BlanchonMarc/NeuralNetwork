from .Layer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd, optim


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, inputs) -> None:
        pass

class SegNet(NeuralNetwork):

    def __init__(self):
        super().__init__(in_channels, n_classes)

        self.layer_1 = SegnetLayer_Encoder(in_channels, 64, 2)
        self.layer_2 = SegnetLayer_Encoder(64, 128, 2)
        self.layer_3 = SegnetLayer_Encoder(128, 256, 3)
        self.layer_4 = SegnetLayer_Encoder(256, 512, 3)
        self.layer_5 = SegnetLayer_Encoder(512, 512, 3)

        self.layer_6 = SegnetLayer_Decoder(512, 512, 3)
        self.layer_7 = SegnetLayer_Decoder(512, 256, 3)
        self.layer_8 = SegnetLayer_Decoder(256, 128, 3)
        self.layer_9 = SegnetLayer_Decoder(128, 64, 2)
        self.layer_10 = SegnetLayer_Decoder(64, n_classes, 2)

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
        output = self.layer_10(inputs=up2, indices=indices_1,
                            output_shape=unpool_shape1, layer_size=2)

        return output
