from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import databases
import os
import importlib
import torch
from utils import transforms3d as t3d
import numpy as np


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
        
        
class template_DataLoader(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.templateDataset(csv_path, data_dir, transforms=trsfm)
        super(template_DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class groundTruth_DataLoader(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, translate=(.01,.01), scale=(0.95,1.05), shear=None, resample=False, fillcolor=0),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
        ])
        trsfm_test = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
        ])
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.groundTruthDataset(csv_path, data_dir, transforms=trsfm)
        super(groundTruth_DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class groundTruth_DataLoader3D(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        rs = np.random.RandomState()
        trsfm_train = [t3d.RandomFlip(rs),t3d.RandomRotate90(rs), t3d.RandomContrast(rs, factor = 0.1, execution_probability=0.25), t3d.ElasticDeformation(rs, 3), t3d.ToTensor(rs), t3d.Normalize(0,1)]
        #trsfm_train = [t3d.RandomFlip(rs),t3d.RandomRotate90(rs),  t3d.ToTensor(rs)]
        trsfm_test = [t3d.ToTensor(rs), t3d.Normalize(0,1)]
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.groundTruthDataset3D(csv_path, data_dir, transforms=trsfm)
        super(groundTruth_DataLoader3D, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class groundTruth_DataLoader64(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, translate=(.01,.01), scale=(0.99,1.01), shear=None, resample=False, fillcolor=0),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]), #scale to [0,1] with a mean of 0.5 and stdev of 0.5
        ])
        trsfm_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.groundTruthDataset(csv_path, data_dir, transforms=trsfm)
        super(groundTruth_DataLoader64, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class groundTruth_DataLoader_upsampled(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10, translate=(.01,.01), scale=(0.95,1.05), shear=None, resample=False, fillcolor=0),
            #transforms.RandomRotation(45),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #standard values?, should be changed based dataset
        ])
        trsfm_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.groundTruthDataset_upsampled(csv_path, data_dir, transforms=trsfm)
        super(groundTruth_DataLoader_upsampled, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
class mnist_DataLoader_upsampled(BaseDataLoader):
    def __init__(self, csv_path, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm_train = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trsfm_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.mnistDataset(csv_path, data_dir, transforms=trsfm)
        super(mnist_DataLoader_upsampled, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)