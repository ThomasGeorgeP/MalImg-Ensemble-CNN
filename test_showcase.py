import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch.nn.functional as F
import torch.nn as nn
from MalImgCNN_Training import MalImgCNN,test_dataset
# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Dataset
# ----------------------------

class_names= [
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


#getting the 3 datasets

transform=transforms.Compose([transforms.Grayscale(1),transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize((0.2860,), (0.3530,))])
test_dataset=torchvision.datasets.ImageFolder("./train/",transform=transform)
test128=test_dataset.t

model_path = "models/fashionmnistmodel0.pth"
model = torch.load(model_path, map_location=DEVICE,weights_only=False)
model.eval()

# ----------------------------
# Image Browser with Probability Bars
# ----------------------------
class ImageBrowser:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.index = 0

        # Figure with two axes: left=image, right=bar chart
        self.fig = plt.figure(figsize=(8, 4))
        self.ax_img = self.fig.add_axes([0.05, 0.1, 0.4, 0.8])  # x0, y0, width, height
        self.ax_bar = self.fig.add_axes([0.55, 0.1, 0.4, 0.8])

        # Buttons
        axprev = plt.axes([0.2, 0.01, 0.1, 0.05])
        axnext = plt.axes([0.7, 0.01, 0.1, 0.05])
        self.bnext = Button(axnext, 'Next')
        self.bprev = Button(axprev, 'Previous')
        self.bnext.on_clicked(self.next_image)
        self.bprev.on_clicked(self.prev_image)

        self.show_image()
        plt.show()

    def show_image(self):
        img, label = self.dataset[self.index]
        img_input = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(img_input)
            probs = F.softmax(output, dim=1)
            _, pred = torch.max(output, 1)

        # ----------------------------
        # Image plot
        # ----------------------------
        self.ax_img.clear()
        self.ax_img.imshow(img.squeeze().cpu(), cmap="gray")
        self.ax_img.set_title(f"Label: {class_names[label]}\nPrediction: {class_names[pred.item()]}")
        self.ax_img.axis('off')

        # ----------------------------
        # Probability bar chart
        # ----------------------------
        self.ax_bar.clear()
        self.ax_bar.barh(class_names, probs[0].cpu().numpy(), color='skyblue')
        self.ax_bar.set_xlim(0, 1)
        self.ax_bar.set_xlabel("Probability")
        self.ax_bar.set_title("Class Probabilities")

        self.fig.canvas.draw()

    def next_image(self, event):
        self.index = (self.index + 1) % len(self.dataset)
        self.show_image()

    def prev_image(self, event):
        self.index = (self.index - 1) % len(self.dataset)
        self.show_image()

# ----------------------------
# Launch GUI
# ----------------------------
browser = ImageBrowser(test_dataset, model)