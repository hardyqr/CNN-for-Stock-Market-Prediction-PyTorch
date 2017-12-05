# Freddy @Blair House
# Nov. 19, 2017

# reference: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from __future__ import print_function, division
import os
from tqdm import *
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#label_table = pd.read_csv('./data/label_table.csv')
#img_name = label_table.ix[,0]


class stock_img_dataset(Dataset):
    """stock canslestick graph dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_table = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_table)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label_table.ix[idx, 1])
        image = io.imread(img_name)
        labels = self.label_table.ix[idx, 2:].as_matrix().astype('int')
        #labels = labels.reshape(-1, 2)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

def show_imgs(image, labels):
    """Show image with landmarks"""
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(2)  # pause a bit so that plots are updated

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #labels = labels * [new_w / w, new_h / h]

        return {'image': img, 'labels': labels}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}


'''test script'''
'''
data = stock_img_dataset('./data/label_table.csv', './data/imgs')
fig = plt.figure()

for i in range(len(data)):
    sample = data[i]

    print(i, sample['image'].shape, sample['labels'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_imgs(**sample)

    #plt.show()
    if i == 3:
        plt.show()
        #os.system('sleep 2s')
        break



scale = Rescale(256)
composed = transforms.Compose([Rescale(256)])


# Apply each of the above transforms on sample.
fig = plt.figure()
sample = data[10]
for i, tsfrm in enumerate([scale, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_imgs(**transformed_sample)

plt.show()
'''
'''data to tensor, and save tensor'''
'''
transformed_dataset = stock_img_dataset(csv_file='./data/label_table.csv',
                                           root_dir='./data/imgs',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['labels'].size())

    if i == 3:
        break
'''
