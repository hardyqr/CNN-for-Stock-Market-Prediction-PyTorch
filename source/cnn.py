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
import math
import time
from tqdm import *

from data_preprocess import *
from utils import *
from logger import Logger

# Hyper Parameters
num_epochs = 12
batch_size = 256
learning_rate = 2e-4

# argv
data_dir = sys.argv[1]
debug = False
load_prev_model = False
direct_test = False
use_gpu = False

if(sys.argv[2] == '1'):
    debug = True
if(sys.argv[3] == '1'):
    load_prev_model = True
if(sys.argv[4] == '1'):
    direct_test = True

if(torch.cuda.is_available()):
    use_gpu = True



''' Data ''' 

# stock Datase
train_set = stock_img_dataset(csv_file=data_dir+'/sample/label_table_train.csv',
        root_dir=data_dir+'/sample/train',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
            ]))
test_set = stock_img_dataset(csv_file=data_dir+'/sample/label_table_test.csv',
        root_dir=data_dir+'/sample/test',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
            ]))
validation_set = stock_img_dataset(csv_file=data_dir+'/sample/label_table_validation.csv',
        root_dir=data_dir+'/sample/validation',
        transform=transforms.Compose([
            Rescale(64),
            ToTensor()
            ]))


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size, 
                                          shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                          batch_size=batch_size, 
                                          shuffle=False)


''' Models '''


# Residual CNN
class res_cnn(nn.Module):
    def __init__(self):
        super(res_cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2))
        #self.fc = nn.Linear(4*4*256, 3)
        self.fc = nn.Linear(4*4*256, 1)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = self.layer6(out4)
        out = self.layer7(out)
        #re_layer_0 = self.layer
        res_layer_1 = self.layer5(self.layer8(out1))
        #print(res_layer.size())
        #print(out.size())
        out5 = res_layer_1+out
        #out5 = out4
        out = self.layer9(out5)
        out = self.layer10(out)
        #out = self.layer10(out)
        out6 = out.view(out.size(0), -1)
        #print(out6.size())
        out7 = self.fc(out6)
        return out7


'''
# GoogLeNet
class google_net(nn.Module):
    def __init__(self):
        super(google_net, self).__init__()
        
        self.conv2d_1x1_a = nn.Conv2d(4,64,kernel_size=1),
        self.conv2d_3x3_a = nn.Conv2d(4,64,kernel_size=3,padding=1),
        self.conv2d_5x5_a = nn.Conv2d(4,64,kernel_size=5,padding=2),
        
        self.conv2d_1x1_b = nn.Conv2d(64,128,kernel_size=1)
        self.conv2d_3x3_b = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv2d_5x5_b = nn.Conv2d(64,128,kernel_size=5,padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        # inception 1


        # inception 2
    return out7
'''



''' train and test ''' 

cnn = res_cnn().double()
if(use_gpu):
    cnn.cuda()

if(load_prev_model):
    print('Loading previous model...')
    cnn.load_state_dict(torch.load('cnn.pkl'))

# Loss and Optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

logger = Logger('./logs')


def test_module(train_size, epoch, data_loader, write=False):
    # Test the Model
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    
    correct_d1 = 0
    correct_d2 = 0
    correct_d3 = 0
    total = 0
    counter = 0
    
    for sample in tqdm(data_loader):
        if(debug and counter >= 3): break
        counter+=1
        
        images = Variable(sample['image'])
        labels = Variable(sample['labels'])
        if(use_gpu):
            images = images.cuda()
            labels = labels.cuda()
        outputs = cnn(images)
        #_, predicted = torch.max(outputs.data, 1)
        
        #labels = to_np(labels.view(-1,1))
        #predicted = np.sign(to_np(outputs.view(-1,1)))
        #print(labels.shape, predicted.shape)

        #=======cpu computation=======#
        # can we migrate it to the GPU?
        labels = to_np(labels)
        outputs = to_np(outputs)
        labels_d1 = labels[:,0]
        #labels_d2 = labels[:,1]
        #labels_d3 = labels[:,2]

        # if output>=0: predict=1, else: predict=-1
        predicted_d1 = np.sign(outputs[:,0])
        #predicted_d2 = np.sign(outputs[:,1])
        #predicted_d3 = np.sign(outputs[:,2])

        # only conduct a prediction when abs(output) > 0.5
        #predicted_high_d1 = np.sign(outputs[:,0])
        #predicted_high_d2 = np.sign(outputs[:,1])
        #predicted_high_d3 = np.sign(outputs[:,2])


        correct_d1 += (predicted_d1 == labels_d1).sum()
        #correct_d2 += (predicted_d2 == labels_d2).sum()
        #correct_d3 += (predicted_d3 == labels_d3).sum()
        
        total += labels.shape[0]
        
        
    acc1 = correct_d1 / total
    #acc2 = correct_d2 / total
    #acc3 = correct_d3 / total
    #acc_total = (correct_d3+correct_d2+correct_d3) / ( total * 3)
    acc2 = 0
    acc3 = 0
    acc_total = acc1
    print('Test Accuracy of the model on the %d test images, Day 11: %.4f %%' % (total, 100 * acc1))
    #print('Test Accuracy of the model on the %d test images, Day 12: %.4f %%' % (total, 100 * acc2))
    #print('Test Accuracy of the model on the %d test images, Day 13: %.4f %%' % (total, 100 * acc3))
    #print('Test Accuracy of the model on the %d test images, total: %.4f %%' % (total, 100 * acc_total))

    test_size = total
    if(write):
        df = pd.DataFrame([[train_size,test_size,acc1,acc2,acc3,acc_total]])
        df.to_csv('./accuracy_records.csv', mode='a',header=False)
    

    #============ TensorBoard logging ============#
    # log validation acc
    if(epoch != -1):
        info = {
                'acc_d1': acc1*100,
                #'acc_d2': acc2*100,
                #'acc_d3': acc3*100,
                #'acc_avg': acc_total*100
                }
        
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)


counter = 0
total = 0
# Train the Model
for epoch in range(num_epochs):
    if(direct_test):
        print('direc _test...')
        test_module(-1, -1,test_loader, False)
        break

    if(debug and counter>=3):
        break
    prev_i = len(train_loader)*epoch

    if(epoch == 0):
        test_module(total, epoch, val_loader,False)

    for i, sample in enumerate(train_loader):
        if(debug and counter>=3):break
        counter+=1
        #if(i == 0): continue
        #print(images,labels)
        images = Variable(sample['image'])
        labels = Variable(sample['labels']).float()
        if(use_gpu):
            images = images.cuda()
            labels = labels.cuda()
        #print(images.size(), labels.size())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images).float()
        #print(outputs.size(), labels.size())
        #print(outputs)
        #print(labels)
        labels = labels[:,0]
        outputs = outputs[:,0]
        loss = criterion(outputs.float(), labels)
        # costly? change this step?
        df = pd.DataFrame([[i+1+prev_i ,to_np(loss)[0]]])
        df.to_csv('./training_loss_records.csv', mode='a',header=False)
        loss.backward()
        optimizer.step()
        
        # costly? change this step?
        total += to_np(labels).shape[0]
        #total += labels.shape[0] # test

        if (i+1) % 1 == 0:
            print ('Epoch [%d/%d], Batch [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs,i+1, math.ceil(len(train_set)/batch_size),loss.data[0]))
            
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
    
    # test at the end of every epoch
    if(epoch + 1 == num_epochs ): 
        # last epoch ends
        print('final test:')
        test_module(total, -1, test_loader, True)
        print('Traine data size: ' + str(total))
    else:
        # during training
        test_module(total, epoch+1,val_loader,False)
    
    # Save the Trained Model
    if(not debug):
        torch.save(cnn.state_dict(), 'cnn.pkl')
    
    # rest 20min for every n epochs
    #rest_time = 1200 #20min
    #n = 10
    #if((epoch+1) % n == 0 and (epoch+1) != num_epochs):
    #    print('Having a rest...')
    #    time.sleep(rest_time)

