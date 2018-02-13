import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


class SegNetBlockUp(nn.Module):
    def __init__(self, in_size, out_size, layer_size):

        super(SegNetBlockUp, self).__init__()
        if layer_size == 2:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        else:
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape, layer_size):

        if layer_size == 2:
            outputs = self.unpool(input=inputs, indices=indices,
                                  output_size=output_shape)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)

        else:
            outputs = self.unpool(input=inputs, indices=indices,
                                  output_size=output_shape)
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)
            outputs = self.conv3(outputs)

        return outputs


class SegNetBlockDown(nn.Module):
    def __init__(self, in_size, out_size, layer_size):

        super(SegNetBlockDown, self).__init__()

        if layer_size == 2:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

        else:
            self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
            self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs, layer_size):

        if layer_size == 2:
            outputs = self.conv1(inputs)
            outputs = self.conv2(outputs)
            unpooled_shape = outputs.size()
            outputs, indices = self.maxpool_with_argmax(outputs)
        else:
            outputs = self.conv1(inputs)
            outputs = self.conv2(outputs)
            outputs = self.conv3(outputs)
            unpooled_shape = outputs.size()
            outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding,
                bias=True, dilation=1):

        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters),
                             kernel_size=k_size, padding=padding,
                             stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
