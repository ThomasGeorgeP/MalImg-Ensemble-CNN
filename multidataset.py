from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms

class MultiScaleDataset(Dataset):
    '''
    THis dataset returns all three resolutions of the same image, so when the dataloader is used
    all three are accessible and can be run by the baselearners
    '''
    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset

        # Storing the transforms
        self.transform128 = transforms.Compose([
                                    transforms.Grayscale(1),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.2860,), (0.3530,))
                                ])
        
        self.transform256 = transforms.Compose([
                                transforms.Grayscale(1),
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.2860,), (0.3530,))
                            ])
        self.transform512 = transforms.Compose([
                                transforms.Grayscale(1),
                                transforms.Resize((512, 512)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.2860,), (0.3530,))
                            ])
        self.transformOG = transforms.Compose([
                                transforms.Grayscale(1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.2860,), (0.3530,))
                            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        
        

        
        img128 = self.transform128(image)
        img256 = self.transform256(image)
        img512 = self.transform512(image)
        imgOG=self.transformOG(image)
        return (img128, img256, img512,imgOG), label
    
