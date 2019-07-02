from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import databases
import os
import importlib
import torch
from utils import transforms3d as t3d
import numpy as np
import importlib
importlib.reload(t3d)


class hdf5_2d_dataloader(BaseDataLoader):
    def __init__(self, hdf5_path, batch_size, shuffle=True, shape = [7,32,32], validation_split=0.0, num_workers=1, training=True):
        trsfm_train = transforms.Compose([
            transforms.Resize(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(45, translate=(.05,.05), scale=(0.95,1.05), shear=None, resample=False, fillcolor=0),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[126.145], std=[31.49]), 
        ])
        trsfm_test = transforms.Compose([
            transforms.Resize(90),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[126.145], std=[31.49]), 
        ])
        
        '''
        means and std for IMPRS dataset
        allDAPI_volume 141.42 18.85
        mask_avgproj 128.16, 0.54
        mask_maxproj 128.51, 1.23
        mask_sumproj 126.145, 31.49
        mask_volume 128.23, 0.84
        '''
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shape = shape
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.hdf5dataset(hdf5_path, shape = self.shape, training = training, transforms=trsfm)
        super(hdf5_2d_dataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        #base data loader requires (dataset, batchsize, shuffle, validation_split, numworkers)

class hdf5_3d_dataloader(BaseDataLoader):
    def __init__(self, hdf5_path, batch_size, shuffle=True, shape = [7,32,32], validation_split=0.0, num_workers=1, training=True):
        rs = np.random.RandomState()
        mean = 141.42
        stdev = 18.85
        trsfm_train = [t3d.shotNoise(rs, alpha = 1.0),
                       #t3d.Downsample(rs, factor = 4.0, order=2),
                       t3d.RandomFlip(rs),
                       t3d.RandomRotate90(rs), 
                       t3d.RandomContrast(rs, factor = 0.2, execution_probability=0.2), 
                       t3d.ElasticDeformation(rs, 3, alpha=20, sigma=3, execution_probability=0.5), 
                       #t3d.GaussianNoise(rs, 3), 
                       #t3d.Normalize(mean, stdev), 
                       t3d.RangeNormalize(),
                       t3d.ToTensor(True)]
        
        #trsfm_train = [t3d.RandomFlip(rs),t3d.RandomRotate90(rs), t3d.ToTensor(rs)]
        trsfm_test = [t3d.shotNoise(rs, alpha = 1.0),
                      #t3d.GaussianNoise(rs, 20),
                      #t3d.Downsample(rs, factor = 4.0, order=2),
                      #t3d.Normalize(mean, stdev), 
                      t3d.RangeNormalize(),
                      t3d.ToTensor(True)]
        
        '''
        means and std for IMPRS dataset
        allDAPI_volume 141.42 18.85
        mask_avgproj 128.16, 0.54
        mask_maxproj 128.51, 1.23
        mask_sumproj 126.145, 31.49
        mask_volume 128.23, 0.84
        allDAPI_volume0701 140.61 17.60
        '''
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shape = shape
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.hdf5dataset(hdf5_path, shape = self.shape, training = training, transforms=trsfm)
        super(hdf5_3d_dataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        #base data loader requires (dataset, batchsize, shuffle, validation_split, numworkers)

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
        #trsfm_train = [t3d.RandomFlip(rs),t3d.RandomRotate90(rs), t3d.RandomContrast(rs, factor = 0.1, execution_probability=0.25), t3d.ElasticDeformation(rs, 3), t3d.ToTensor(rs)]
        trsfm_train = [t3d.RandomFlip(rs),t3d.RandomRotate90(rs), t3d.ToTensor(rs)]
        trsfm_test = [t3d.ToTensor(rs)]
        
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