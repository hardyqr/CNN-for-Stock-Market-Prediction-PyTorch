# Freddy @DC, uWaterloo, ON, Canada
# Nov 13, 2017

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd
#from tqdm import *
from data_preprocess import *
from utils import *

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# stock Datase
print('loading trainning data...')
train_set = stock_img_dataset(csv_file='./data/label_table_train.csv',
        root_dir='./data/train',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
            ]))
print('loading testing data...')
test_set = stock_img_dataset(csv_file='./data/label_table_test.csv',
        root_dir='./data/test',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
            ]))
validation_set = stock_img_dataset(csv_file='./data/label_table_validation.csv',
        root_dir='./data/validation',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
            ]))


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=test_set,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                          batch_size=batch_size, 
                                          shuffle=False)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2))


        self.fc = nn.Linear(8*8*16, 3)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        res_layer = self.layer5(out1)
        #print(res_layer.size())
        #print(out4.size())
        out5 = res_layer+out4
        out6 = out5.view(out5.size(0), -1)
        #print(out6.size())
        out7 = self.fc(out6)
        return out7
        
cnn = CNN().double()
cnn.cuda()

# Loss and Optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

loss_records = []

# Train the Model
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
        #if(i == 0): continue
        #print(images,labels)
        images = Variable(sample['image']).cuda()
        labels = Variable(sample['labels']).float().cuda()
        #print(images.size(), labels.size())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images).float()
        #print(outputs.size(), labels.size())
        #print(outputs)
        #print(labels)
        loss = criterion(outputs.float(), labels)
        loss_records.append(loss)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_set)//batch_size, loss.data[0]))


# save training records (loss)
loss_records = pd.DataFrame(np.array(loss_records))
loss_records.to_csv('./training_loss_records.csv')

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for sample in test_loader:
    images = Variable(sample['image']).cuda()
    labels = Variable(sample['labels']).cuda()
    outputs = cnn(images)
    #_, predicted = torch.max(outputs.data, 1)
    labels = labels.view(-1,1).data.numpy()
    predicted = np.sign(outputs.view(-1,1).data.numpy())
    #print(labels.shape, predicted.shape)
    total += labels.shape[0]
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the %d test images: %d %%' % (total, 100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')

