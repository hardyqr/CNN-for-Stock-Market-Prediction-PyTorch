# Freddy @BH
# Dec 20, 2017

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
num_epochs = 10
batch_size = 256
#learning_rate = 1e-3
learning_rate = 1e-4

# argv
#data_dir = sys.argv[1]

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

class DigitDataset(Dataset):

    def __init__(self, matrix_csv_path, label_csv_path, dtype):
        matrix_data = pd.read_csv(matrix_csv_path,header=None)
        label_data = pd.read_csv(label_csv_path,header=None)
        self.dtype = dtype
        
        #labels = np.ravel(label_data.ix[:,1:3]) # np.ravel: return a flattened array
        labels = label_data.ix[:,1:3]
        pixels = np.array(matrix_data.ix[:,1:])

        self.N = int(pixels.shape[0]/5) # N: num of train samples
        if(self.N != labels.shape[0]):
            print('matrix & label dimension mismatch')
        pixels = pixels.reshape(self.N,5,1,20)
        pixels = pixels.transpose(0,2,3,1)

        #self.pixels_train = np.array(pixels).reshape([self.N,1,20,5])
        self.pixels_train = pixels
        self.labels_train = np.array(labels).reshape(self.N,3)
        
    def __getitem__(self,index):
        label = torch.from_numpy(self.labels_train[index]).type(self.dtype)
        img = torch.from_numpy(self.pixels_train[index]).type(self.dtype)
        return img, label

    def __len__(self):
        return self.N


data_root = sys.argv[1]

train_path = data_root+'/matrix_train.csv'
train_label_path = data_root+'/label_train.csv'
test_path = data_root+'/matrix_test.csv'
test_label_path = data_root+'/label_test.csv'
val_path = data_root+'/matrix_validation.csv'
val_label_path = data_root+'/label_validation.csv'

dtype = torch.FloatTensor

training_dataset = DigitDataset(train_path, train_label_path,dtype)
train_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle=True)

test_dataset = DigitDataset(test_path, test_label_path, dtype)
test_loader = DataLoader(test_dataset)

val_dataset = DigitDataset(val_path, val_label_path,dtype)
val_loader = DataLoader(val_dataset)

print(len(training_dataset))
print(len(test_dataset))
print(len(val_dataset))


''' Models '''

def helper(x):
    if(x>=0.5):
        return 1.
    else: 
        return 0.


# CNN

# reproduction of CLEAR-Trade:
# https://arxiv.org/pdf/1709.01574.pdf
class res_cnn(nn.Module):
    def __init__(self):
        super(res_cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=1),
            nn.LeakyReLU(0.3))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 2,kernel_size=1),
            nn.LeakyReLU(0.3))
        self.pl = nn.AvgPool2d((20,5))
        self.sm = nn.Softmax()

    def forward(self, x):
        #print(x.size())
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer35(out)
        out = self.layer4(out)
        #print(out.size())
        out = self.pl(out)
        #print(out.size())
        out = out.view(out.size(0),-1)
        #print(out.size())
        out = self.sm(out)
        return out


''' train and test ''' 

cnn = res_cnn().double()
if(use_gpu):
    cnn.cuda()

if(load_prev_model):
    print('Loading previous model...')
    cnn.load_state_dict(torch.load('cnn4matrix.pkl'))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
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
        

        images = Variable(sample[0]).double()
        labels = Variable(sample[1]).float()

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
        scores = to_np(outputs)
        labels_d1 = labels[:,0]
        #labels_d1_onehot[np.arrage(labels_d1.shape[0]), labels_d1] = 1
        #labels_d2 = labels[:,1]
        #labels_d3 = labels[:,2]


        # if output>=0: predict=1, else: predict=-1
        #predicted_d1 = list(map(helper,outputs[:,0]))
        predicted_d1 = scores.argmax()
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
    print('Test Accuracy of the model on the %d test images, Day 21: %.4f %%' % (total, 100 * acc1))
    #print('Test Accuracy of the model on the %d test images, Day 22: %.4f %%' % (total, 100 * acc2))
    #print('Test Accuracy of the model on the %d test images, Day 23: %.4f %%' % (total, 100 * acc3))
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
        images = Variable(sample[0]).double()
        labels = Variable(sample[1]).float()
        if(use_gpu):
            images = images.cuda()
            labels = labels.cuda()
        #print(images.size(), labels.size())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        score = cnn(images).float()
        labels = labels[:,0]
        #outputs = outputs[:,0]
        
        loss = criterion(score, labels.long())
        #df = pd.DataFrame([[i+1+prev_i ,to_np(loss)[0]]])
        #df.to_csv('./training_loss_records_4matrix.csv', mode='a',header=False)
        loss.backward()
        optimizer.step()
        
        # costly? change this step?
        total += to_np(labels).shape[0]
        #total += labels.shape[0] # test


        #============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'loss': loss.data[0]
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, i+1+prev_i)

        '''
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in cnn.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), i+1+prev_i)
            logger.histo_summary(tag+'/grad', to_np(value.grad), i+1+prev_i)
        '''
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Batch [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs,i+1, math.ceil(len(training_dataset)/batch_size),loss.data[0]))

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
        torch.save(cnn.state_dict(), 'cnn4matrix.pkl')
