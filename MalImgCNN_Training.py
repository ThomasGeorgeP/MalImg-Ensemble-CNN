import torch
from torch.utils.data import DataLoader 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import pathlib
from torch.optim.lr_scheduler import ReduceLROnPlateau

malimg_classes = [
    'Adialer.C',
    'Agent.FYI',
    'Allaple.A',
    'Allaple.L',
    'Alueron.gen!J',
    'Autorun.K',
    'C2LOP.P',
    'C2LOP.gen!g',
    'Dialplatform.B',
    'Dontovo.A',
    'Fakerean',
    'Instantaccess',
    'Lolyda.AA1',
    'Lolyda.AA2',
    'Lolyda.AA3',
    'Lolyda.AT',
    'Malex.gen!J',
    'Obfuscator.AD',
    'Rbot!gen',
    'Skintrim.N',
    'Swizzor.gen!E',
    'Swizzor.gen!I',
    'VB.AT','Wintrim.BX','Yuner.A']

class MalImgCNN(nn.Module):
    def __init__(self, layers=[32,64,128,256], dropout_rate=0.2):
        super().__init__()
        previous_channels = 1  # grayscale input (even the images were b/w when stored in dir, linux processes it as RGB)
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
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Globally average the 32x32 maps
            nn.Flatten(),
            nn.Linear(previous_channels, 512), # previous_channels is 256
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 25) # num_classes is 25
        )
    def forward(self, x):
        x= self.network(x)
        return self.classifier(x)
    
if __name__=="__main__":
    model_dir=pathlib.Path("./model/")
    model_dir.mkdir(exist_ok=True,parents=True)

    param_path=model_dir/"parameters.pth"
    model_path=model_dir/"model.pth"

    #printing versions for future reference

    IMAGE_SIZE=(128,128)
    BATCH_SIZE=32
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

    transform=transforms.Compose([transforms.Grayscale(1),transforms.Resize(IMAGE_SIZE),transforms.ToTensor(),transforms.Normalize((0.2860,), (0.3530,))])
    train_dataset=torchvision.datasets.ImageFolder("./train/",transform=transform)

    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=8) #8 out of 16 is good for my pc, change if needed

    test_dataset=torchvision.datasets.ImageFolder("./test/",transform=transform)
    test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=8)

    try:
        model=MalImgCNN().to(DEFAULT_DEVICE)
        model.load_state_dict(torch.load(param_path,weights_only=True,map_location=DEFAULT_DEVICE))  #random initialization gives better op when ensembling
        print("Model Loaded")
    except:
        model=MalImgCNN().to(DEFAULT_DEVICE)

    optimizer=optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-05)
    criterion=nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,)
    epochs=100

    for epoch in range(epochs):

        avg_train_loss=0
        model.train()
        for batchx,batchy in train_dataloader:
            batchx, batchy = batchx.to(DEFAULT_DEVICE), batchy.to(DEFAULT_DEVICE)
            
            optimizer.zero_grad()

            loss=criterion.forward(model(batchx),batchy)
            avg_train_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        model.eval()
        total_val_loss=0
        with torch.no_grad():
            for batchx_val, batchy_val in test_dataloader:
                batchx_val, batchy_val = batchx_val.to(DEFAULT_DEVICE), batchy_val.to(DEFAULT_DEVICE)
                
                val_outputs = model(batchx_val)
                val_loss = criterion(val_outputs, batchy_val)
                total_val_loss += val_loss.item()
                print(model.network(batchx_val).shape)

        avg_val_loss = total_val_loss / len(test_dataloader)
        avg_train_loss/=len(train_dataloader)
        print(f"Epoch {epoch}: Loss: {loss}")
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        
        scheduler.step(avg_val_loss)
        if epoch%3==0:
            torch.save(model.state_dict(),param_path,)
            torch.save(model,model_path)

    #saving the model


    torch.save(model.state_dict(),param_path)
    torch.save(model,model_path)