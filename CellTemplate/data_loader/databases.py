from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageEnhance
from utils import transforms3d
import h5py

class hdf5dataset(Dataset):
    def __init__(self, h5_path, shape = (7,32,32), training = True, transforms=None):
        st = pd.HDFStore(h5_path)
        self.noise = False
        if training:
            self.data = st['train_data'].values
            self.label = st['train_labels'].values
        else:
            self.data = st['test_data'].values
            self.label = st['test_labels'].values
        self.transforms = transforms
        self.data_len = self.data.shape[0]
        self.shape = shape
        
        count_labels = {}
        for item in self.label:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        weights2 = []
        for key in count_labels:
            weights.append(1. / count_labels[key])
            weights2.append(count_labels[key])
        
        weightsnp = np.asarray(weights)
        weights2np = np.asarray(weights2)
        maxnum = np.amax(weights2np)
        weightsnp = weightsnp*maxnum
        self.weight = torch.FloatTensor(weightsnp)
            
        
    def __getitem__(self, index):
        #TODO: validate this algorithm --> particularly make sure the reshape matches
        # in java, we do for slice, for x, for y --> slice changes slowest
        num_pixels = 1
        for dim in self.shape: num_pixels = num_pixels*dim
        img = self.data[index, 0:num_pixels]
        img = np.reshape(img, self.shape, order = 'C') #last index of shape changes fastest
        img = img.astype(float)
        label = self.label[index] - 1
        
        if self.noise:
            img = np.random.poisson(lam=img, size=None)
        
        
        #img_as_img = Image.fromarray(img) #convert to PIL
        if self.transforms is not None:
            if len(self.shape) > 2:
                for transform in self.transforms:
                    img = transform(img)
                img_as_tensor = img
                #img_as_tensor = image.permute(0,2,1,3) #from 1x7x32x32 to 1x32x7x32
                #img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
                img_as_tensor = img_as_tensor.type(torch.FloatTensor)
            else:
                image = img.astype(float)
                img_as_img = Image.fromarray(image)
                img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
        '''
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
        '''
        
    
    def __len__(self):
        return self.data_len

class templateDataset(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.csv_data = pd.read_csv(csv_path, header = 0)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.csv_data)
        
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.csv_data.iloc[index, 7]) #path data is in the 7th column
        image = io.imread(img_name)
        img_as_img = Image.fromarray(image) #convert to PIL
        label = self.csv_data.iloc[index, 6]  #label is in the 6th column
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len
class groundTruthDataset(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.csv_data = pd.read_csv(csv_path, header = 0, engine='python')
        self.csv_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.csv_data)
        all_labels = self.csv_data.iloc[:,1]
        count_labels = {}
        for item in all_labels:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        weights2 = []
        for key in count_labels:
            weights.append(1. / count_labels[key])
            weights2.append(count_labels[key])
        
        weightsnp = np.asarray(weights)
        weights2np = np.asarray(weights2)
        maxnum = np.amax(weights2np)
        weightsnp = weightsnp*maxnum
        self.weight = torch.FloatTensor(weightsnp)
        
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, str(self.csv_data.iloc[index, 0])) #path data is in the 1rst column
        image = io.imread(img_name)
        image = image.astype(float)
        #print(image.shape)
        img_as_img = Image.fromarray(image) #convert to PIL
        label = int(self.csv_data.iloc[index, 1])  #label is in the 2nd column
        #if label == 3: label = 2
        label = label - 1 # go from 1,2 to 0,1 --> not great solution
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len
    
class groundTruthDataset3D(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.csv_data = pd.read_csv(csv_path, header = 0, engine='python')
        self.csv_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.csv_data)
        all_labels = self.csv_data.iloc[:,1]
        count_labels = {}
        for item in all_labels:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        weights2 = []
        for key in count_labels:
            weights.append(1. / count_labels[key])
            weights2.append(count_labels[key])
        
        weightsnp = np.asarray(weights)
        weights2np = np.asarray(weights2)
        maxnum = np.amax(weights2np)
        weightsnp = weightsnp*maxnum
        self.weight = torch.FloatTensor(weightsnp)
        
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                str(self.csv_data.iloc[index, 0])) #path data is in the 1rst column
        image = io.imread(img_name)
        image = image.astype(float)
        #print(image.shape) 7x32x32
        label = int(self.csv_data.iloc[index, 1])  #label is in the 2nd column
        label = label - 1 # go from 1,2 to 0,1 --> not great solution
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)
            img_as_tensor = image
            #img_as_tensor = image.permute(0,2,1,3) #from 1x7x32x32 to 1x32x7x32
            #img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
            img_as_tensor = img_as_tensor.type(torch.FloatTensor)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len
    
class groundTruthDataset_upsampled(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.csv_data = pd.read_csv(csv_path, header = 0, engine='python')
        self.csv_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.csv_data)
        all_labels = self.csv_data.iloc[:,1]
        count_labels = {}
        for item in all_labels:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        for key in count_labels:
            weights.append(1. / count_labels[key])
        self.weight = torch.FloatTensor(weights)
        
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                str(self.csv_data.iloc[index, 0])) #path data is in the 1rst column
        image = io.imread(img_name)
        img_as_img = Image.fromarray(image) #convert to PIL
        img_as_img = img_as_img.resize((60,60), resample=Image.NEAREST)
        img_as_img = img_as_img.resize((100,100), resample=Image.NEAREST)
        #img_as_img = img_as_img.resize((150,150), resample=Image.NEAREST)
        img_as_img = img_as_img.resize((256,256), resample=Image.NEAREST)
        rgbimg = Image.new("RGB", img_as_img.size, color=(0,0,0))
        rgbimg.paste(img_as_img)
        values =  list(rgbimg.getdata())
        new_image= rgbimg.point(lambda argument: argument*1)
        label = int(self.csv_data.iloc[index, 1])  #label is in the 2nd column
        label = label - 1 # go from 1,2 to 0,1 --> not great solution
        if self.transforms is not None:
            img_as_tensor = self.transforms(new_image)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len

class mnistDataset(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.data = torch.load(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.data[1])
        
    def __getitem__(self, index):
        img = self.data[0][index]
        trans = transforms.ToPILImage()
        img_as_img = trans(img)
        img_as_img = img_as_img.resize((60,60), resample=Image.NEAREST)
        img_as_img = img_as_img.resize((100,100), resample=Image.NEAREST)
        #img_as_img = img_as_img.resize((150,150), resample=Image.NEAREST)
        img_as_img = img_as_img.resize((256,256), resample=Image.NEAREST)
        rgbimg = Image.new("RGB", img_as_img.size, color=(0,0,0))
        rgbimg.paste(img_as_img)
        values =  list(rgbimg.getdata())
        new_image= rgbimg.point(lambda argument: argument*1)
        label = self.data[1][index]
        if self.transforms is not None:
            img_as_tensor = self.transforms(new_image)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len
    