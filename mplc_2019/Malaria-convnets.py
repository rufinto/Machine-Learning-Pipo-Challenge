# =============================================================================
# WE ARE IMPORTING ALL NEEDING MODULES
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CREATING DATASET
# =============================================================================

# 1- transform

'''
because our images is not in same shape, we have to reshape its by apply different transform:
    Resize all images to shape (100, 100, 3),
    Crop all of them to size 100.
'''

data_transform = transforms.Compose([
           transforms.Resize((100, 100)),
           transforms.RandomCrop(100),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

# 2- importing data    

'''
we have two image's classes :
    parasitized with index_class 1,
    no_parasitized with index_class 0.
'''

cell_Dataset = datasets.ImageFolder(root='.\cell_images', transform=data_transform) 

# 3- creating of train and test set

'''
in generally train set takes 80% of our dataset and test set, the rest. 
so we create our splits like this.
'''

train_len = int(0.8*len(cell_Dataset)) # train size
test_len = len(cell_Dataset) - train_len # test size
train, test = random_split(cell_Dataset, (train_len, test_len)) # we split dataset in train and test
print('dataset size:', len(cell_Dataset))
print('train dataset size:', len(train)) 
print('test dataset size:', len(test))

train_set = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True) # load train set
test_set = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True) # load test set

classes = ('No_Parasitized', 'Parasitized') # our classes

# =============================================================================
#   WE SHOW SOME TRAINING IMAGES
# =============================================================================

'''
becuse we have convert our images to tensor, we have to reconvert them to numpy images,
in other to show its. 
'''

# fonction to reconvert tensor images to numpy

def imshow(image):
    image = image / 2 + 0.5 # unormalize
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
        
# we get some random training images
    
plt.figure(figsize=(15, 6))
for i in range(10):
    dataiter = iter(train_set) 
    image, label = dataiter.__next__() # choose ramdomly one image
    plt.subplot(2, 5,i+1)
    # show images and labels
    imshow(make_grid(image))
    plt.title('classe:{}'.format(classes[label]))
plt.show()

# =============================================================================
# CREATING MODEL (CNN)
# =============================================================================

class Net(nn.Module): # creating our layers
    
    def __init__(self):
       super(Net, self).__init__()
       
       # we initialize full connected layers of our neural network
       
       self.conv1 = nn.Conv2d(3, 6, 5) # first conv layer with 6 filters of size 5x5
       self.pool = nn.MaxPool2d(2, stride=2) # maxpool with filter size 2x2
       self.conv2 = nn.Conv2d(6, 16, 5) # second conv layer with 16 filters of size 5x5
       self.fc1 = nn.Linear(16*22*22, 128) # first full connected layer
       self.fc2 = nn.Linear(128, 64) # second full connected layer
       self.fc3 = nn.Linear(64, 32) # third full connected layer
       self.fc4 = nn.Linear(32, 2) # fourth full connected layer
      
       # we make xavier initialisation to avoid death weights in our relu (activate function) 
       
    def weights_init(m):
        if isinstance(m, nn.conv2d):
            nn.init.xavier_uniform_(m.weight.data, nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(m.bias.data, nn.init.calculate_gain('relu'))
       
        # we define our forward pass
        
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
        
net = Net() # name of our model

'''
To have maximum power in computing gradients, we have to execute our model on gpu.
So here verify if our pc our pc is cuda compatible gpu.
'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('cuda is availible for this GPU')
    print('runing on GPU ...')
else:
    print('cuda can not be executable on this gpu')
    print('running on CPU ...')
    device = torch.device("cpu")

net.to(device)

# =============================================================================
#  WE DEFINE LOSS FONCTION
# =============================================================================

criterion = nn.CrossEntropyLoss() # combine log_softmax and nn_loss

# =============================================================================
#   CREATING OPTIMIZER
# =============================================================================

optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))

# =============================================================================
# TRAINING NETWORK
# =============================================================================

EPOCHS = 4
loss_history = [] # to save loss valu after each iteration

# we train our model 

print('starting training...')
print('average loss after each batch of 3000 images')

for epoch in range(EPOCHS):
    
    running_loss = 2
    for i, data in enumerate(train_set, 0):
        
        # X is inputs, Y is labels. data is a list of [inputs, labels]
        
        X, Y = data[0].to(device), data[1].to(device) # to put our data in gpu if cuda is availible
        
        # to put all gradient to zero after each epoch
        
        net.zero_grad() 
        
        # forward pass, backward pass and optimizer
        
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        
        # we want to see loss decrease after 100 mini-batches
        
        running_loss += loss.item()
        
        # we print loss after a batch of 2000 images for each bach
        
        if i % 3000 == 2999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            loss_history.append(running_loss)
            running_loss = 0.0
            
print('finished Training')  
 
# we plot loss decrease over time  
     
plt.figure(figsize=(8, 8))
plt.plot(range(len(loss_history)), loss_history, c='r', ls='-', lw=2)
plt.xlabel('time')
plt.ylabel('loss')
plt.title('loss_decrease') 
plt.show()

# =============================================================================
# WE SE PERFORMANCE OF NETWORK ON TRAIN SET AND TEST SET
# =============================================================================

correct_train = 0
correct_test = 0
total_train = 0
total_test = 0
Y_train_true = []
Y_train_pred = []
Y_test_true = []
Y_test_pred = []

with torch.no_grad():
    for data in train_set:
        X, Y = data[0].to(device), data[1].to(device)
        output = net(X)
        Y_train_true.append(Y)
        pred = torch.argmax(output)
        Y_train_pred.append(pred)
        if pred == Y:
            correct_train += 1
        total_train += 1
print('train_Accuracy = {}%'.format(round(correct_train/total_train, 3)*100)) # to have accuracy on train set
print('confusion_matrix:', confusion_matrix(Y_train_true, Y_train_pred)) # to see confusion matrix on train set


with torch.no_grad():
    for data in test_set:
        X, Y = data[0].to(device), data[1].to(device)
        output  = net(X)
        Y_test_true.append(Y)
        pred = torch.argmax(output)
        Y_test_pred.append(pred)
        if pred == Y:
            correct_test += 1
        total_test += 1
print('test_Accuracy = {}%'.format(round(correct_test/total_test, 3)*100)) # to have accuracy on test set
print('confusion_matrix:', confusion_matrix(Y_test_true, Y_test_pred)) # to see confusion matrix on test set
      
# =============================================================================
# HERE WE SAVE ALL PARAMETERS OF OUR TRAINING MODEL 
# =============================================================================

PATH = '.\save_params'
torch.save(net.state_dict(), PATH)

print('program parameters save to:', PATH)
             
# =============================================================================
# WE SHOW PREDICTION OF NETWORK ON SOME TEST IMAGES
# =============================================================================

# 1- we get some random test images and we predict each class

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


print('finish executing.')
















































