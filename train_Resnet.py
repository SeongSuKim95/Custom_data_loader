import torch

import torch.nn as nn

from torch.utils.data.dataloader import DataLoader

import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.optim as optim
from torchvision.transforms.transforms import Resize

from Dataset import CatsAndDogsDataset

## Network

class CNN(nn.Module):
    def __init__(self,in_channels = 3 , num_classes = 2):
        super(CNN,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), stride = (1,1), padding = (1,1))

        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.fc = nn.Linear(32 * 8 * 8 , num_classes)
    
    def forward(self,x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)        
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.reshape(x.shape[0],-1) # x.shape[0] --> batch_size
        #print(x.shape)
        x = self.fc(x)

        return x

## Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

## Transforms

transform = transforms.Compose(
            [
            transforms.Resize((64,64)),            
            transforms.ToTensor()
            ]
                              )

## Hyperparams

in_channels = 3
learning_rate = 0.001
batch_size = 32
num_classes = 2
num_epochs = 10

## Dataset, DataLoader
# 1. Load dataset from custom dataset
# 2. Split into train, test set
# 3. Make Dataloader 

Dataset = CatsAndDogsDataset("/home/sungsu21/Project/data/dogs_cats_data/train/","/home/sungsu21/Project/data/dogs_cats_data/train_csv.csv",transform = transform)
train_set, test_set = torch.utils.data.random_split(Dataset,[20000,5000])

train_loader = DataLoader(dataset = train_set, shuffle = True, batch_size = batch_size)
test_loader = DataLoader(dataset = test_set, shuffle = True, batch_size = batch_size)

## Initializing network

model = CNN(in_channels = 3, num_classes = num_classes).to(device)

## Optimizer, criterion
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):    
    for batch_idx, (data,targets) in enumerate(train_loader):
        
        # Get data to cuda
        data = data.to(device = device)
        targets = targets.to(device = device)

        scores = model(data)
        loss  = criterion(targets,scores)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def Check_Accuracy(loader,model):
    
    if loader.dataset.train:
        print("Checking accuaracy on train data")
    else:
        print("Checking accuaracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader :
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            # scores : 64 X 10-->> Find maximum probability class
            # scores.max(1) --> (values, indices)
            _, predictions = scores.max(1) # We are interested in indices 

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy  {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()

Check_Accuracy(train_loader,model)
Check_Accuracy(test_loader,model)       