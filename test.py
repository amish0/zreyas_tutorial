# from tkinter.ttk import _Padding
import torch
from torch import nn
model_1 = torch.nn.Sequential(torch.nn.Conv2d(1,20,(5,3), padding=2), # 28x28 -> 32x32 -> 30x30
          torch.nn.ReLU(),
        #   torch.nn.Conv2d(20,64,5, padding=2),
        #   torch.nn.ReLU()
        )


model = torch.nn.Sequential(torch.nn.Conv2d(3,20,5), # [1, 20, 252, 252]
                            torch.nn.BatchNorm2d(20), # [1, 20, 252, 252]
                            torch.nn.ReLU(), # [1, 20, 252, 252]
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



