import argparse
import os
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


class EncoderBlock2:
    def __init__(self):
        print('Encoder Block2')
    def forward():
        print('Forward Encoder Block2')

class EncoderBlock3:
    def __init__(self):
        print('Encoder Block3')
    def forward():
        print('Forward Encoder Block3')

class DecoderBlock2:
    def __init__(self):
        print('Decoder Block2')
    def forward():
        print('Forward Decoder Block2')

class DecoderBlock3:
    def __init__(self):
        print('Decoder Block3')
    def forward():
        print('Forward Decoder Block3')


class Segnet:
    def __init__(self):
        #super().__init__()
        print('In Segnet')
        self.enc1 = EncoderBlock2()
        self.enc2 = EncoderBlock2()
        self.enc3 = EncoderBlock3()
        self.enc4 = EncoderBlock3()
        self.enc5 = EncoderBlock3()

        self.dec1 = DecoderBlock3()
        self.dec2 = DecoderBlock3()
        self.dec3 = DecoderBlock3()
        self.dec4 = DecoderBlock2()
        self.dec5 = DecoderBlock2()

if __name__ == '__main__':
    Segnet()


'''
class Encoder:
    def __init__(self, arg, size , input_var):
        self.size = size
        self.arg = arg
        if self.arg:
            self.nature = self.Block3()
        else:
            self.nature = self.Block4()

        self.Operate(input_var)



    def Block3(self):

        self.first_conv = nn.Conv2d(self.size, self.size, kernel_size=3, padding=1)
        self.first_batch  = nn.BatchNorm2d(self.size, momentum= .1)
        self.second_conv = nn.Conv2d(self.size, self.size, kernel_size=3, padding=1)
        self.second_batch = nn.BatchNorm2d(self.size, momentum= .1)

    def Block4(self):
        self.first_conv = nn.Conv2d(self.size, self.size, kernel_size=3, padding=1)
        self.first_batch  = nn.BatchNorm2d(self.size, momentum= .1)
        self.second_conv = nn.Conv2d(self.size, self.size, kernel_size=3, padding=1)
        self.second_batch = nn.BatchNorm2d(self.size, momentum= .1)
        self.third_conv = nn.Conv2d(self.size, self.size / 2, kernel_size=3, padding=1)
        self.third_batch = nn.BatchNorm2d(self.size / 2,  momentum= .1)

    def Operate(self,input_var):


        first_layer = torch.nn.functional.relu(self.first_batch(self.first_conv(input_var)))
        intermediate_layer = torch.nn.functional.relu(self.second_batch(self.second_conv(first_layer)))


        if not self.arg:
            intermediate_layer = torch.nn.functional.relu(self.third_batch(self.third_conv(intermediate_layer)))

        last_layer , unused = torch.nn.functional.max_pool2d(intermediate_layer,kernel_size=2, stride=2,return_indices=True)

        print('everything went ok')

        return last_layer




class Decoder:
    def __init__(self, arg):
        if arg:
            self.nature = self.Block3()
        else:
            self.nature = self.Block4()

    def Block3(self):
        print('This is a Block 3')

    def Block4(self):
        print('This is a Block 4')
'''
