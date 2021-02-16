import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


train_dataset = datasets.MNIST(root = "/home/sungsu21/Project/data/", train = True, transform = transforms.ToTensor(),download = False)

train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)

def get_mean_std(loader):

    # VAR[X] = E[X**2] - E[X]**2

    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        
        channels_sum += torch.mean(data,dim=[0,2,3])
        channels_squares_sum += torch.mean(data**2,dim=[0,2,3])
        num_batches +=1
    
    mean = channels_sum / num_batches
    std = (channels_squares_sum/num_batches - mean ** 2) **0.5

    return mean, std

mean, std = get_mean_std(train_loader)
print(mean)
print(std)