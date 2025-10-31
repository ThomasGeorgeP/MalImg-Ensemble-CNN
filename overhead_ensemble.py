import torch
from torch.utils.data import DataLoader 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import pathlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MalImgCNN_Training import MalImgCNN
import torch.nn.functional as F
from multidataset import MultiScaleDataset
from sys import exit

class EnsembleANN(nn.Module):
    """
A simple ANN meta-learner for ensembling model predictions.

This class defines a "stacked" ensemble model. It is a small
neural network that takes the output predictions (logits or
probabilities) from multiple other models as its input.

It then learns to weigh these predictions to produce a single,
combined, and hopefully more accurate final classification.

Args:
    input_size (int): The total number of input features.
        (e.g., num_models * num_classes_per_model).
    num_classes (int): The number of final output classes.
    hidden_size (int, optional): The number of neurons in the
        hidden layer.
"""
    def __init__(self,layers:list[int]=[128,64],droupout:float=0.5,):
        super().__init__()
        self.DEFAULT_DEVICE=("cuda" if cuda.is_available() else "cpu")
        CNN512=MalImgCNN()
        CNN256=MalImgCNN()
        CNN128=MalImgCNN()

        try:
            CNN512.load_state_dict(torch.load("model/parameters512.pth",map_location=self.DEFAULT_DEVICE,weights_only=True))
            CNN256.load_state_dict(torch.load("model/parameters256.pth",map_location=self.DEFAULT_DEVICE,weights_only=True))
            CNN128.load_state_dict(torch.load("model/parameters128.pth",map_location=self.DEFAULT_DEVICE,weights_only=True))
        except:
            print("Couldnt Load all Base Learners, Please Ensure weights are at model/parameters512,256,128")
            exit()

        
        self.baselearners = nn.ModuleList([CNN512, CNN256, CNN128])
        #inputsize fixed at 768 (256*3 256 from each CNN)
        previous_size=768
        self.layers=[]
        for i in layers:
            self.layers.append(nn.Linear(previous_size,i))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(i))
            self.layers.append(nn.Dropout(droupout))
            previous_size=i
        self.layers.append(nn.Linear(previous_size,25))

        self.network=nn.Sequential(*self.layers)
        #self.transform512=transforms.Resize((512,512))
    def forward(self,x:list[torch.Tensor]):
        self.baselearners.eval()
        x512=x[0].to(device=self.DEFAULT_DEVICE)
        x256=x[1].to(device=self.DEFAULT_DEVICE)
        x128=x[2].to(device=self.DEFAULT_DEVICE)
        with torch.no_grad():
            y512=self.baselearners[0].network(x512)
            y256=self.baselearners[1].network(x256)
            y128=self.baselearners[2].network(x128)

        y512=torch.flatten(F.adaptive_avg_pool2d(y512,(1,1)),start_dim=1)
        y256=torch.flatten(F.adaptive_avg_pool2d(y256,(1,1)),start_dim=1)
        y128=torch.flatten(F.adaptive_avg_pool2d(y128,(1,1)),start_dim=1)

        input_layer=torch.cat([y512,y256,y128],dim=1)
        return self.network(input_layer)

if __name__=="__main__":
    BATCH_SIZE=8
    DEFAULT_DEVICE=("cuda" if cuda.is_available() else "cpu")

    train_image_folder=torchvision.datasets.ImageFolder("train/")

    train_dataset=MultiScaleDataset(train_image_folder)

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,pin_memory=True,num_workers=8,shuffle=True)

    test_image_folder=torchvision.datasets.ImageFolder("test/")

    test_dataset=MultiScaleDataset(test_image_folder)

    test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,pin_memory=True,num_workers=8,shuffle=True)

    

    model=EnsembleANN().to(DEFAULT_DEVICE)

    try:
        model.load_state_dict(torch.load("model/ensembleparams.pth",map_location=DEFAULT_DEVICE,weights_only=True))
        print("Model Loaded! ")
    except:
        pass
    optimizer=optim.Adam(model.parameters(),lr=0.00003,weight_decay=1e-05)
    criterion=nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,)

    epochs=100

    for i in range(epochs):
        total_training_loss=0
        model.train()
        for batchx,batchy in train_dataloader:
            
            batchy=batchy.to(DEFAULT_DEVICE)
            optimizer.zero_grad()

            loss=criterion.forward(model.forward(batchx),batchy)
            loss.backward()
            total_training_loss+=loss.item()

            optimizer.step()
        else:

            total_training_loss/=len(train_dataloader)
            model.eval()
            with torch.no_grad():
                test_loss=0
                for batchx,batchy in test_dataloader:

                    batchy=batchy.to(DEFAULT_DEVICE)
                    loss=criterion.forward(model.forward(batchx),batchy)
                    test_loss+=loss.item()

                test_loss/=len(test_dataloader)
                scheduler.step(test_loss)

                print(f"Epoch {i+1}: Training Loss: {total_training_loss}.  Test Loss: {test_loss}")
                if i%3==0:
                    torch.save(model.state_dict(),"model/ensembleparams.pth")
                    torch.save(model,"model/ensemblemodel.pth")

    