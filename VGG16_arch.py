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

x = torch.randn(5,3,224,224)
print(x.shape)
y = model(x)
print(y.shape)

# dataloader loader
# train_data_loader
# validation_data_loader
train_data_loader=[]
valid_data_loader=[]
# optimizer
import torch.optim as optim
optm = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

epach_no = 10
batch_size = 8
validation_loss_old = inf
counter_var = 0
early_stopping = 10
reduce_lr_initializer = 5
reduce_lr_counter = 0
for epoch in range(epach_no):
    model.train() #
    for x_train, y_train in train_data_loader:
        y_batch = model(x_train)
        loss = loss_fn(y_batch, y_train)
        print(f'Epoch {epoch} Loss: {loss.item()}')
        optm.zero_grad()
        loss.backward()
        optm.step()
    model.eval() #
    loass_val = 0
    for x_val, y_val in valid_data_loader:
        y_val_batch = model(x_val)
        loss = loss_fn(y_val_batch, y_val)
        loass_val += loss.item()
    print(f'Epoch {epoch} Loss: {loass_val/len(valid_data_loader)}')

    # Early stopping
    if loass_val < validation_loss_old:
        counter_var = 0
        reduce_lr_counter = 0
        torch.save(model.state_dict(), 'model_best.pth')
        validation_loss_old = loass_val
    else:
        counter_var += 1
        if counter_var > early_stopping:
            break
        if reduce_lr_counter % reduce_lr_initializer == 0:
            # reduce learning rate
            optm = optim.Adam(model.parameters(), lr=0.001)
            reduce_lr_counter += 1
    torch.save(model.state_dict(), 'model_iteration_'+str(epoch)+'.pth')
    
# save model

# define data loader
# train the model
# write inference script

