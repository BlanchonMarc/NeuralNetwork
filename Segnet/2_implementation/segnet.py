import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import autograd
import numpy as np


class SegNet(nn.Module):

    def __init__(self, input_nbr, label_nbr):
        super(SegNet, self).__init__()

        self.batchNorm_momentum = 0.1
        self.kernel_size = 3
        self.conv_padding = 1

        self.maxPooling_KernelSize = 2

        self.conv = []
        self.batch = []

        self.conv_length = [2, 2, 3, 3, 3, 3, 3, 3, 2, 1]

        self.type = ['encoder'] * 5
        self.type += ['decoder'] * 5
        print(self.type)

        #############
        # ENCODER
        #############
        # Block2, encoder, input_size -> 64
        self.block1 = self.Block(input_nbr, 64, self.conv_length[0],
        self.type[0])

        # Block2, encoder, 64 -> 128
        self.block2 = self.Block(64, 128, self.conv_length[1],
        self.type[1])

        # Block3, encoder, 128 -> 256
        self.block3 = self.Block(128, 256, self.conv_length[2],
        self.type[2])

        # Block3, encoder, 256 -> 512
        self.block4 = self.Block(256, 512, self.conv_length[3],
        self.type[3])

        # Block3, encoder, 512 -> 512
        self.block5 = self.Block(512, 512, self.conv_length[4],
        self.type[4])

        ###############
        # DECODER
        ###############

        # Block3, decoder, 512 -> 512
        self.block6 = self.Block(512, 512, self.conv_length[5],
        self.type[5])

        # Block3, decoder, 512 -> 256
        self.block7 = self.Block(512, 256, self.conv_length[6],
        self.type[6])

        # Block3, decoder, 256 -> 128
        self.block8 = self.Block(256, 128, self.conv_length[7],
        self.type[7])

        # Block2, decoder, 128 -> 64
        self.block9 = self.Block(128, 64, self.conv_length[8],
        self.type[8])

        # Block2, decoder, 64 -> label_nbr
        self.block10 = self.Block(64, label_nbr, self.conv_length[9],
        self.type[9])

        print(self.conv)

    def Block(self, input_size, output_size, block_size, identifier):

        for i in range(0, block_size):

            if block_size == 1 and identifier == 'decoder':
                self.conv.append(nn.Conv2d(input_size, input_size,
                kernel_size=self.kernel_size,
                padding=self.conv_padding))

                self.batch.append(nn.BatchNorm2d(input_size,
                momentum=self.batchNorm_momentum))

                self.conv.append(nn.Conv2d(input_size, output_size,
                kernel_size=self.kernel_size,
                padding=self.conv_padding))

            else:
                if identifier == 'encoder':
                    if i > 0:
                        self.conv.append(nn.Conv2d(output_size,
                        output_size, kernel_size=self.kernel_size,
                        padding=self.conv_padding))

                    else:
                        self.conv.append(nn.Conv2d(input_size,
                        output_size, kernel_size=self.kernel_size,
                        padding=self.conv_padding))

                    self.batch.append(nn.BatchNorm2d(output_size,
                    momentum=self.batchNorm_momentum))

                else:
                        if i > block_size - 2:
                            self.conv.append(nn.Conv2d(input_size,
                            output_size, kernel_size=self.kernel_size,
                            padding=self.conv_padding))

                            self.batch.append(nn.BatchNorm2d(output_size,
                            momentum=self.batchNorm_momentum))
                        else:
                            self.conv.append(nn.Conv2d(input_size,
                            input_size, kernel_size=self.kernel_size,
                            padding=self.conv_padding))

                            self.batch.append(nn.BatchNorm2d(input_size,
                            momentum=self.batchNorm_momentum))

        print(len(self.conv))

    def forward(self, x):
        storing_firstInput = x
        print(len(self.conv))
        store_input = x
        store_start = 0
        store_maxpoolindex = []
        tracker = 1
        for ind in range(0, len(self.conv_length)):

            store_end = store_start + self.conv_length[ind]

            if self.type[ind] == 'decoder':

                print('index maxpool : ', (ind - tracker))
                store_input = F.max_unpool2d(store_input,
                store_maxpoolindex[ind - tracker],
                    kernel_size=self.maxPooling_KernelSize, stride=2)
                tracker = tracker + 2
                print('tracker', tracker)

            for indx in range(store_start, store_end):
                store_input = F.relu(self.batch[indx](
                    self.conv[indx](store_input)))
                print(self.conv[indx])

            if self.type[ind] == 'encoder':

                store_input, index = F.max_pool2d(store_input,
                kernel_size=self.maxPooling_KernelSize, stride=2,
                return_indices=True)
                store_maxpoolindex.append(index)
                print('Index Size', index.size())
                print('len, storage index', len(store_maxpoolindex))

            print(store_input.size())
            store_start = store_end

        return F.softmax(store_input, dim=2)


# Execution
batch_size = 1
input_size = 8
num_classes = 8

nb = 64
input = autograd.Variable(torch.rand(batch_size, input_size, nb, nb))


model = SegNet(input_nbr=input_size, label_nbr=num_classes)
out = model(input)

print('out', out)
