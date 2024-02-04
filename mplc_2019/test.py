import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

data_transform = transforms.Compose([
           transforms.Resize((100, 100)),
           transforms.RandomCrop(100),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

cell_Dataset = datasets.ImageFolder(root='.\cell_images', transform=data_transform) 
train_len = int(0.8*len(cell_Dataset))
test_len = len(cell_Dataset) - train_len
train, test = random_split(cell_Dataset, (train_len, test_len))
train_set = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
test_set = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
classes = ('No_Parasitized', 'Parasitized')

class Net(nn.Module):
    
    def __init__(self):
       super(Net, self).__init__()
       # we initialize full connected layers of our neural network
       self.conv1 = nn.Conv2d(3, 6, 5)# output 220x220x6
       self.pool = nn.MaxPool2d(2, stride=2)# output 110x110X6
       self.conv2 = nn.Conv2d(6, 16, 5)#output 116x116x16 #in_channel=profondeur de couleur de l'image en entrée, out_channel=nbre de filtre
       self.fc1 = nn.Linear(16*22*22, 128)
       self.fc2 = nn.Linear(128, 64)
       self.fc3 = nn.Linear(64, 32)
       self.fc4 = nn.Linear(32, 2)
      
    def weights_init(m):
        if isinstance(m, nn.conv2d):
            nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(m.bias.data, nn.init.calculate_gain('relu'))
       
    def forward(self, X):
        X = self.pool(f.relu(self.conv1(X)))
        X = self.pool(f.relu(self.conv2(X)))
        #print(X.shape)
        X = X.view(-1, 16*22*22)
        X = f.relu(self.fc1(X))
        X = f.relu(self.fc2(X))
        X = f.relu(self.fc3(X))
        X = self.fc4(X)
        return X

net = Net()
PATH = '.\save_params'
net.load_state_dict(torch.load(PATH))


def imshow(image):
    image = image / 2 + 0.5 # unormalize
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))

plt.figure(figsize=(20, 100))
for i in range(100):
    with torch.no_grad():
        plt.subplot(25, 4, i+1)
        dataiter = iter(test_set)
        image, label = dataiter.__next__()
        output = net(image)
        predict = torch.argmax(output)
        imshow(make_grid(image))
        plt.title('classe:{}, predict:{}'.format(classes[label], classes[predict]))
plt.show()


















