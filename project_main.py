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

#printing versions for future reference


# print(f'''  Versions
# Torch: {torch.__version__}
# Torchvision: {torchvision.__version__}
# Numpy: {np.__version__}
# Matplotlib: {matplotlib.__version__}
# ''')

if cuda.is_available():

    DEFAULT_DEVICE="cuda"
    cuda.manual_seed_all(120)
    torch.manual_seed(120)

else:

    DEFAULT_DEVICE="cpu"
    torch.manual_seed(120)

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2860,), (0.3530,)),transforms.Resize((512,512))])
train_dataset=torchvision.datasets.ImageFolder("./train/",transform=transform)

train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True,pin_memory=True,num_workers=8) #8 out of 16 is good for my pc, change if needed

for batchx,batchy in train_dataloader:
    print(batchx,batchy)
    break