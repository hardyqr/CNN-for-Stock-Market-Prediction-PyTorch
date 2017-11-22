# Freddy @DC, uWaterloo, ON, Canada
# Nov 13, 2017

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from data_preprocess import *
from utils import *

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# stock Dataset
train_set = stock_img_dataset(csv_file='./data/label_table_train.csv',
        root_dir='./data/train',
        transform=transforms.Compose([
            Rescale(256),
            ToTensor()
            ]))
test_set = stock_img_dataset(csv_file='./data/label_table_test.csv',
        root_dir='./data/test',
        transform=transforms.Compose([
            Rescale(256),
            ToTensor()
            ]))
validation_set = stock_img_dataset(csv_file='./data/label_table_validation.csv',
        root_dir='./data/validation',
        transform=transforms.Compose([
            Rescale(256),
            ToTensor()
            ]))


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size, 
                                          shuffle=False)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 3)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
cnn = CNN().double()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
        #if(i == 0): continue
        #print(images,labels)
        images = Variable(sample['image']).double()
        labels = Variable(sample['labels']).double()
        print(images.size(), labels.size())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')

