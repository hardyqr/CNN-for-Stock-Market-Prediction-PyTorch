# Freddy @DC, uWaterloo, ON, Canada
# Nov 13, 2017

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd
import sys
from tqdm import *
from data_preprocess import *
from utils import *
from logger import Logger

# Hyper Parameters
num_epochs = 10
batch_size = 100
learning_rate = 2e-3

# argv
data_dir = sys.argv[1]
debug = False
if(sys.argv[2] == '1'):
    debug = True

# stock Datase
train_set = stock_img_dataset(csv_file=data_dir+'/sample/label_table_train.csv',
        root_dir=data_dir+'/sample/train',
        transform=transforms.Compose([
            Rescale(128),
            ToTensor()
            ]))
test_set = stock_img_dataset(csv_file=data_dir+'/sample/label_table_test.csv',
        root_dir=data_dir+'/sample/test',
        transform=transforms.Compose([
            Rescale(128),
            ToTensor()
            ]))
validation_set = stock_img_dataset(csv_file=data_dir+'/sample/label_table_validation.csv',
        root_dir=data_dir+'/sample/validation',
        transform=transforms.Compose([
            Rescale(128),
            ToTensor()
            ]))


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=test_set,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                          batch_size=batch_size, 
                                          shuffle=False)



# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2))


        self.fc = nn.Linear(16*16*128, 3)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = self.layer6(out4)
        out = self.layer7(out)
        res_layer = self.layer5(self.layer8(out1))
        #print(res_layer.size())
        #print(out.size())
        out5 = res_layer+out
        #out5 = out4
        out6 = out5.view(out5.size(0), -1)
        #print(out6.size())
        out7 = self.fc(out6)
        return out7
        
cnn = CNN().double()
if(torch.cuda.is_available()):
    cnn.cuda()

# Loss and Optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

logger = Logger('./logs')

counter = 0
total = 0
# Train the Model
for epoch in range(num_epochs):
    prev_i = len(train_loader)*epoch
    for i, sample in enumerate(train_loader):
        if(debug and counter>=3):break
        counter+=1
        #if(i == 0): continue
        #print(images,labels)
        images = Variable(sample['image'])
        labels = Variable(sample['labels']).float()
        if(torch.cuda.is_available()):
            images = images.cuda()
            labels = labels.cuda()
        #print(images.size(), labels.size())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images).float()
        #print(outputs.size(), labels.size())
        #print(outputs)
        #print(labels)
        loss = criterion(outputs.float(), labels)
        df = pd.DataFrame([i+1+prev_i ,to_np(loss)])
        df.to_csv('./training_loss_records.csv', mode='a',header=False)
        loss.backward()
        optimizer.step()
        
        total += to_np(labels).shape[0]

        if (i+1) % 1 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_set)//batch_size, loss.data[0]))
            
            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
            'loss': loss.data[0]
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, i+1+prev_i)

            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in cnn.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), i+1+prev_i)
                logger.histo_summary(tag+'/grad', to_np(value.grad), i+1+prev_i)


print('traine data size: ' + str(total))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
counter = 0
for sample in tqdm(test_loader):
    if(debug and counter >= 3): break
    counter+=1

    images = Variable(sample['image'])
    labels = Variable(sample['labels'])
    if(torch.cuda.is_available()):
        images = images.cuda()
        labels = labels.cuda()
    outputs = cnn(images)
    #_, predicted = torch.max(outputs.data, 1)
    labels = to_np(labels.view(-1,1))
    predicted = np.sign(to_np(outputs.view(-1,1)))
    #print(labels.shape, predicted.shape)
    total += labels.shape[0]
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the %d test images: %.4f %%' % (total, 100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')

