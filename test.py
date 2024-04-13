# from tkinter.ttk import _Padding
import torch
from torch import nn
model_1 = torch.nn.Sequential(torch.nn.Conv2d(1,20,(5,3), padding=2), # 28x28 -> 32x32 -> 30x30
                            # 28x28 -> 34x34 -> 32x32 Eith Padding\
                            # Kernal Size = (5,5), padding= 5//2 = 2, 28x28 -> 28x28
                            # kernal Size = (3,3), padding = 3//2 = 1, 28x28 -> 28x28
                            # kernel size = (5,3), padding = (5//2,3//2) = (2,1), 28x28 -> 32x30
                            # without padding
                            # 28x28 -> 26x26 (kernal size = (3,3))
                            # 28x28 -> 24x24 (kernal size = (5,5))
                            # Kernal size =(5,3), 28x28 -> 24x26
          torch.nn.ReLU(),
        #   torch.nn.Conv2d(20,64,5, padding=2),
        #   torch.nn.ReLU()
        )
# print(model_1)


# input image size = 256x256
# batch Size = 1
# Channel size = 3
# input dimension = 1x3x256x256 [batch size, channel size, height, width]
model = torch.nn.Sequential(torch.nn.Conv2d(3,20,5), # [1, 20, 252, 252]
                            torch.nn.BatchNorm2d(20), # [1, 20, 252, 252]
                            torch.nn.ReLU(), # [1, 20, 252, 252]
                            # torch.nn.Conv2d(20,64,5), # [1, 64, 248, 248]
                            # torch.nn.ReLU() # [1, 64, 248, 248]
                            torch.nn.Conv2d(20,64,3), # [1, 64, 250, 250]
                            torch.nn.BatchNorm2d(64), # [1, 64, 250, 250]
                            torch.nn.ReLU(), # [1, 64, 250, 250]
                            torch.nn.Conv2d(64,256,7,padding=3), # [1, 256, 250, 250] 7//2 = 3
                            torch.nn.ReLU(), # [1, 256, 250, 250]
                            torch.nn.Conv2d(256,64,(5,3), padding=(2,1)), # [1, 64, 246, 248], 5//2 = 2, 3//2 = 1
                            torch.nn.BatchNorm2d(64), # [1, 64, 246, 248]
                            ) 
# Path: test.py

x = torch.randn(1,3,256,256)
print(x.shape)
y = model(x)
print(y.shape)



