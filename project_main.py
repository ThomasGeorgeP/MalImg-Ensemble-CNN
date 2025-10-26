import torch
from torch.utils.data import Dataset,DataLoader 
import torchvision
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import pathlib
from torch.optim.lr_scheduler import ReduceLROnPlateau


model_dir=pathlib.Path("./model/")
model_dir.mkdir(exist_ok=True,parents=True)

param_path=model_dir/"parameters.pth"
model_path=model_dir/"model.pth"

#printing versions for future reference

IMAGE_SIZE=(512,512)
BATCH_SIZE=8
# print(f'''  Versions
# Torch: {torch.__version__}
# Torchvision: {torchvision.__version__}
# Numpy: {np.__version__}
# Matplotlib: {matplotlib.__version__}
# ''')

if cuda.is_available():

    DEFAULT_DEVICE="cuda"
    cuda.manual_seed_all(69)
    torch.manual_seed(69)

else:

    DEFAULT_DEVICE="cpu"
    torch.manual_seed(120)

transform=transforms.Compose([transforms.Grayscale(1),transforms.ToTensor(),transforms.Normalize((0.2860,), (0.3530,)),transforms.Resize(IMAGE_SIZE)])
train_dataset=torchvision.datasets.ImageFolder("./train/",transform=transform)

train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=8) #8 out of 16 is good for my pc, change if needed

test_dataset=torchvision.datasets.ImageFolder("./test/",transform=transform)
test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=8)


class MalImgCNN(nn.Module):
    def __init__(self, layers=[32,64,128,256], dropout_rate=0.2):
        super().__init__()
        previous_channels = 1  # grayscale input
        convolutional_layers = []

        for out_channels in layers:
            convolutional_layers.append(nn.Conv2d(
                in_channels=previous_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            convolutional_layers.append(nn.LeakyReLU())
            convolutional_layers.append(nn.Dropout2d(dropout_rate))
            convolutional_layers.append(nn.BatchNorm2d(out_channels))
            convolutional_layers.append(nn.MaxPool2d(2))

            previous_channels = out_channels

        self.network = nn.Sequential(*convolutional_layers)

    def forward(self, x):
        return self.network(x).flatten(start_dim=1)
try:
    model=MalImgCNN().to(DEFAULT_DEVICE)
    #model.load_state_dict(torch.load(param_path,weights_only=True,map_location=DEFAULT_DEVICE))  random initialization gives better op when ensembling
    print("Model Loaded")
except:
    model=MalImgCNN().to(DEFAULT_DEVICE)

optimizer=optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-05)
criterion=nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
epochs=100

for epoch in range(epochs):
    for batchx,batchy in train_dataloader:
        batchx, batchy = batchx.to(DEFAULT_DEVICE), batchy.to(DEFAULT_DEVICE)
        model.train()
        optimizer.zero_grad()

        loss=criterion.forward(model(batchx),batchy)
        loss.backward()

        optimizer.step()
    model.eval()
    t_output,t_label=next(iter(test_dataloader))
    t_output,t_label=t_output.to(DEFAULT_DEVICE),t_label.to(DEFAULT_DEVICE)
    loss=criterion.forward(model(t_output),t_label)
    scheduler.step(loss.item())
    print(f"Epoch {epoch}: Loss: {loss}")
    if epoch%3==0:
        torch.save(model.state_dict(),param_path,)
        torch.save(model,model_path)

#saving the model


torch.save(model.state_dict(),param_path)
torch.save(model,model_path)