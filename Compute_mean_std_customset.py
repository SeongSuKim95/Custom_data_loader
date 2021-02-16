import torch
import torchvision.transforms as transforms
import os
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
import PIL.Image as Image
from PIL import ImageStat
from torchvision.transforms.transforms import PILToTensor, ToPILImage
from Dataloader import Breast_cancer_dataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
transform = transforms.Compose([    
                                transforms.ToPILImage(),
                                transforms.Resize((56,56), interpolation = 2),
                                transforms.PILToTensor(), 
                                #transforms.ToTensor()
                                ])
data_df = pd.read_csv("./train_csv.csv")
print(len(data_df))

Dataset = Breast_cancer_dataset(df_data = data_df, transform = transform)
#Dataset = torchvision.datasets.CIFAR10(root='/home/sungsu21/Project/data',train=True, download=False, transform = transform)

data_num = len(Dataset)
loader = torch.utils.data.DataLoader(Dataset,batch_size = data_num, shuffle = False, num_workers = 0)

step = len(loader)
square_1 = 0
square_2 = 0
square_3 = 0

for i, (images,_) in enumerate(loader):  
    images = images.float().to(device)
    #images *= 255
    for i in range(data_num):
        Temp = images[i,:,:,:]
        square_1 += torch.mean(torch.square(Temp[0,:,:]))
        square_2 += torch.mean(torch.square(Temp[1,:,:]))
        square_3 += torch.mean(torch.square(Temp[2,:,:]))
    square_1 = square_1/data_num
    square_2 = square_2/data_num
    square_3 = square_3/data_num

    mean = torch.mean(images,dim = 0)

    mean_1 = torch.mean(mean[0,:])
    mean_2 = torch.mean(mean[1,:])
    mean_3 = torch.mean(mean[2,:])

    std_1 = torch.sqrt(square_1 - mean_1**2)
    std_2 = torch.sqrt(square_2 - mean_2**2)
    std_3 = torch.sqrt(square_3 - mean_3**2)
    
    DATA_mean = [mean_1, mean_2, mean_3] 
    DATA_std = [std_1,std_2,std_3]

print(DATA_mean, DATA_std)

# CIFAR 10
# mean[0.4914,0.4822,0.4465], std[0.2470,0.2435,0.2616]

# Breast-cancer dataset 
# mean[0.7335, 0.6343, 0.8133], std[0.1444, 0.2013, 0.1346]