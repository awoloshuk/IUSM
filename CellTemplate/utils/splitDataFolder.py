#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:19:12 2019

@author: andre
"""

#python3 splitDataFolder.py -r "../data/" -c "../data/GroundTruth_052219/Label_1.csv" -s 0.1
import argparse
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
    folder_path = root_dir
    print(folder_path)
    
    images = list_files(folder_path)

    print("Total number of images = " + str(len(images)))
    random.seed(0)
    idx = list(range(len(images)))
    random.shuffle(idx)
    test_num = int(split_fraction*len(images))
    test_idx = idx[0:test_num-1]
    train_idx = idx[test_num:len(images)-1]
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
        #shutil.move(old_image_path, new_image_path)
        count = count + 1
    
    count = 0    
    for i in train_idx:
        image = images[i]
        filename = csv_data.iloc[i, 0]
        label = csv_data.iloc[i, 1]
        train_list.append([filename, label])
        folder_name = "Train"
        new_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print("New Folder " + folder_name + " created")
    
        old_image_path = os.path.join(folder_path, filename)
        new_image_path = os.path.join(new_path, filename)
        #shutil.move(old_image_path, new_image_path)
        count = count + 1
    
    print("Training images = " + str(len(train_list)))
    print("Test images = " + str(len(test_list)))
    cols = ['Filename', 'Label']
    csv_Test = pd.DataFrame(test_list, columns = cols) 
    csv_Train = pd.DataFrame(train_list, columns = cols)
    
    csv_Test.to_csv(path_or_buf = rootd+"/Test.csv", index=False)
    csv_Train.to_csv(path_or_buf = rootd+"/Train.csv", index=False)

    

#rootd = "/Users/andre/Desktop/CellTemplate/data/"
#csv = "/Users/andre/Desktop/CellTemplate/data/groundTruth/Train/Label_1.csv"

def main(rootd, csv, split):
    splitTrainTest(rootd, csv, split)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--root', default=None, type=str,
                        help='root directory file path (default: None)')
    parser.add_argument('-c', '--csv2', default=None, type=str,
                        help='path to csv (default: None)')
    parser.add_argument('-s', '--split', default=0.1, type=float,
                        help='split percentage for test, e.g. 0.1')
    args = parser.parse_args()
    
    
    
    if args.root:
        rootd = args.root    
    if args.csv2:
        csv1 = args.csv2
    if args.split:
        split1 = args.split
        
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c Train/Train.csv', for example.")
    main(rootd, csv1, split1)
