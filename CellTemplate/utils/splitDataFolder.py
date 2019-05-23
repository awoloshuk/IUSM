#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:19:12 2019

@author: andre
"""

import os, os.path, shutil
import pathlib
import random
import pandas as pd
import numpy as np

def list_dirs(directory):
    """Returns all directories in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_dir()]


def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and f.name.endswith('.tiff')]


def splitTrainTest(root_dir, csv_path, split_fraction):
    t = "Train/"
    folder_path = os.path.join(root_dir, t)
    print(folder_path)
    
    images = list_files(folder_path)

    print(len(images))
    random.seed(0)
    idx = list(range(len(images)))
    random.shuffle(idx)
    test_num = int(split_fraction*len(images))
    test_idx = idx[0:test_num]
    train_idx = idx[test_num+1:]
    csv_data = pd.read_csv(csv_path, header = 0, engine='python')
    test_list = [[]]
    train_list = [[]]
    count = 0
    for i in test_idx:
        image = images[i]
        filename = csv_data.iloc[i, 0]
        label = csv_data.iloc[i, 1]
        test_list.append([filename, label])
        #edit file name 
        #add to new data list 
        
        
        folder_name = "Test"
        new_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print("New Folder " + folder_name + " created")
    
        old_image_path = os.path.join(folder_path, filename)
        new_image_path = os.path.join(new_path, filename)
        shutil.move(old_image_path, new_image_path)
        count = count + 1
        
    for i in train_idx:
        image = images[i]
        filename = csv_data.iloc[i, 0]
        label = csv_data.iloc[i, 1]
        train_list.append([filename, label])
        
    cols = ['Filename', 'Label']
    csv_Test = pd.DataFrame(test_list, columns = cols) 
    csv_Train = pd.DataFrame(train_list, columns = cols)
    
    csv_Test.to_csv(path_or_buf = rootd+"Test/Test.csv", index=False)
    csv_Test.to_csv(path_or_buf = rootd+"Train/Train_generated.csv", index=False)
        
rootd = "/Users/andre/Desktop/CellTemplate/data/"
csv = "/Users/andre/Desktop/CellTemplate/data/Train/Label_1.csv"

splitTrainTest(rootd, csv, 0.1)

