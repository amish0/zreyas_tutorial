from calendar import c
from math import inf
from tkinter.messagebox import NO
from unittest import loader
import torch
from torch import nn

from yolov5.train import train
# Input image type: RGB C = 3
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, kernel_size=(3,3), padding=(1,1)), # 224x224 -> 224x224
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)), # 224x224 -> 224x224
    torch.nn.ReLU(),
    # Stage 1
    torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 224x224 -> 112x112
    torch.nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)), # 112x112 -> 112x112
    torch.nn.ReLU(),
    torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=(1,1)), # 112x112 -> 112x112
    torch.nn.ReLU(),
    # Stage 2
    torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 112x112 -> 56x56
    torch.nn.Conv2d(128, 256, kernel_size=(3,3), padding=(1,1)), # 56x56 -> 56x56
    torch.nn.ReLU(),
    torch.nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)), # 56x56 -> 56x56
    torch.nn.ReLU(),
    torch.nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)), # 56x56 -> 56x56
    torch.nn.ReLU(),
    # Stage 3
    torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 56x56 -> 28x28
    torch.nn.Conv2d(256, 512, kernel_size=(3,3), padding=(1,1)), # 28x28 -> 28x28
    torch.nn.ReLU(),
    torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)), # 28x28 -> 28x28
    torch.nn.ReLU(),
    torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)), # 28x28 -> 28x28
    torch.nn.ReLU(),
    # Stage 4
    torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 28x28 -> 14x14
    torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)), # 14x14 -> 14x14
    torch.nn.ReLU(),
    torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)), # 14x14 -> 14x14
    torch.nn.ReLU(),
    torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)), # 14x14 -> 14x14
    torch.nn.ReLU(),
    torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1)), # 14x14 -> 14x14
    torch.nn.ReLU(),
    # Stage 5
    torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # 14x14 -> 7x7 # output no of feature
    torch.nn.Flatten(),
    torch.nn.Linear(512*7*7, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 1000),
    torch.nn.Softmax(dim=1)

)
