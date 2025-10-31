import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch.nn.functional as F
import torch.nn as nn
from overhead_ensemble import EnsembleANN 
from multidataset import MultiScaleDataset 
import random 
import sys
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class_names = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J',
    'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A',
    'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3',
    'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N',
    'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A'
]
try:
    test_image_folder = torchvision.datasets.ImageFolder("val/")
except:
        print("Please download and extract the folders from https://drive.google.com/file/d/1hoSiS0YXXU6yfDYdbpYKCD-10QCjF1jr/view?usp=sharing")
        sys.exit()
test_dataset = MultiScaleDataset(test_image_folder)

model_path = "model/ensembleparams_final.pth"
model = EnsembleANN().to(device=DEVICE)
try:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,weights_only=True))
except Exception as e:
    print(f"Error: Could not load model weights: {e}")
    print("Please ensure model parameters are in model/ensembleparams_final.pth")
    exit()
model.eval()



class ImageBrowser:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        
       
        self.num_images = len(self.dataset)
        self.shuffled_indices = list(range(self.num_images))
        random.shuffle(self.shuffled_indices) #shuffle indexes instead of using dataloader
        
        self.list_position = 0

        self.fig = plt.figure(figsize=(16, 8))
        self.ax_img = self.fig.add_axes([0.05, 0.1, 0.4, 0.8]) 
        self.ax_bar = self.fig.add_axes([0.55, 0.1, 0.4, 0.8])
        plt.title("Malware Classification")
       
        axprev = plt.axes([0.2, 0.01, 0.1, 0.05])
        axnext = plt.axes([0.7, 0.01, 0.1, 0.05])
        self.bnext = Button(axnext, 'Next')
        self.bprev = Button(axprev, 'Previous')
        self.bnext.on_clicked(self.next_image)
        self.bprev.on_clicked(self.prev_image)

        self.show_image()
        plt.show()

    def show_image(self):
        
        index = self.shuffled_indices[self.list_position] #picks the shuffled index


        img_, label = self.dataset[index]
        
        img_to_show = img_[3]


        model_input = [i.unsqueeze(0) for i in img_] #unsqueezing cause the baseCNN takes 4D

        with torch.no_grad():
            output = self.model(model_input)
            probs = F.softmax(output, dim=1)
            _, pred = torch.max(output, 1)

        self.ax_img.clear()
        self.ax_img.imshow(img_to_show.squeeze().cpu(), cmap="gray")
        self.ax_img.set_title(f"Label: {class_names[label]}\nPrediction: {class_names[pred.item()]}")
        self.ax_img.axis('off')


        self.ax_bar.clear()
        self.ax_bar.barh(class_names, probs[0].cpu().numpy(), color='skyblue')
        self.ax_bar.set_xlim(0, 1)
        self.ax_bar.set_xlabel("Probability")
        self.ax_bar.set_title("Class Probabilities")

        self.fig.canvas.draw()

    def next_image(self, event):
       
        self.list_position = (self.list_position + 1) % self.num_images
        self.show_image()

    def prev_image(self, event):
       
        self.list_position = (self.list_position - 1) % self.num_images
        self.show_image()

browser = ImageBrowser(test_dataset, model)


wrong_count=0
for batchx,batchy in test_dataset:
    batchx=[i.unsqueeze(0) for i in batchx]
    output=model.forward(batchx)
    probability=nn.functional.softmax(output,dim=1)
    _,prediction=torch.max(probability,1)

    if prediction!=batchy:
        wrong_count+=1
print(f"Wrong count: {wrong_count} Out of {len(test_dataset)}")