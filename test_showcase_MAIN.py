import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch.nn.functional as F
import torch.nn as nn
from overhead_ensemble import EnsembleANN  # Make sure these are imported
from multidataset import MultiScaleDataset  # Make sure these are imported
import random  # <-- 1. Import random

# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Dataset
# ----------------------------

class_names = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J',
    'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A',
    'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3',
    'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N',
    'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A'
]

test_image_folder = torchvision.datasets.ImageFolder("val/")

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


# ----------------------------
# GUI Class
# ----------------------------
class ImageBrowser:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        
        # --- 2. CREATE AND SHUFFLE THE INDICES ---
        self.num_images = len(self.dataset)
        self.shuffled_indices = list(range(self.num_images))
        random.shuffle(self.shuffled_indices)
        
        self.list_position = 0  # <--- 3. Use this to track position in the *shuffled list*

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
        # --- 4. GET THE SHUFFLED INDEX ---
        # Get the real dataset index from our shuffled list
        index = self.shuffled_indices[self.list_position]

        # img_ is a tuple: (img128, img256, img512)
        # Each tensor is 3D: (1, H, W)
        img_, label = self.dataset[index]
        
        # We'll display the 512px image, which is img_[2]
        img_to_show = img_[2]

        # Create the model input:
        # 1. Re-order to [512, 256, 128]
        # 2. Add a batch dimension (.unsqueeze(0)) to each 3D tensor
        model_input = [
            img_[0].unsqueeze(0),  # img512 -> shape (1, 1, 512, 512)
            img_[1].unsqueeze(0),  # img256 -> shape (1, 1, 256, 256)
            img_[2].unsqueeze(0)   # img128 -> shape (1, 1, 128, 128)
        ]

        with torch.no_grad():
            output = self.model(model_input)
            probs = F.softmax(output, dim=1)
            _, pred = torch.max(output, 1)

        # ----------------------------
        # Image plot
        # ----------------------------
        self.ax_img.clear()
        self.ax_img.imshow(img_to_show.squeeze().cpu(), cmap="gray")
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
        # --- 5. UPDATE THE LIST POSITION ---
        self.list_position = (self.list_position + 1) % self.num_images
        self.show_image()

    def prev_image(self, event):
        # --- 5. UPDATE THE LIST POSITION ---
        self.list_position = (self.list_position - 1) % self.num_images
        self.show_image()

# ----------------------------
# Launch GUI
# ----------------------------
browser = ImageBrowser(test_dataset, model)