import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import json
import math

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def visualizeDataset(dataloader):
    '''
    Visualize a batch of tensors
    '''
    images, labels = next(iter(dataloader))
    plt.imshow(torchvision.utils.make_grid(images, nrow=8).permute(1, 2, 0))
    
def visualizeBatch(dataloader, normalized):
    '''
    Visualize all the images in a batch in a subplot
    Visualize one image as its own figure
    '''
    images, labels = next(iter(dataloader))
    img = unnormTensor(images[0], normalized)
    plt.imshow(img)
    fig = plt.figure(figsize=(40, 40))
    batch = math.ceil(math.sqrt(dataloader.batch_size))
    for i in range(len(images)):
        a = fig.add_subplot(batch,batch,i+1)
        img = unnormTensor(images[i], normalized)
        imgplot = plt.imshow(img) #have to unnormalize data first!
        plt.axis('off')
        a.set_title("Label = " +str(labels[i].numpy()), fontsize=30)

def unnormTensor(tens, normalized):
    '''
    Takes a image tensor and returns the un-normalized numpy array scaled to [0,1]
    '''
    mean = [0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]
    img = tens.permute(1,2,0).numpy()
    if normalized: 
        img = img*std + mean
    if img.shape[2] == 1:
        img = img.squeeze()
    img = (img + abs(np.amin(img))) / (abs(np.amin(img))+abs(np.amax(img)))
    return img

def visualizationOutGray(data, output, target, classes, normalized):
    '''
    Used to show the first test image in a batch with its label and prediction
    Data size is batch_size, 1, 28, 28 (grayscale images!)
    '''
    ig = plt.figure()
    output_cpu = output.to(torch.device("cpu"))
    target_cpu = target.to(torch.device("cpu"))
    output_idx = (np.argmax(output_cpu[0], axis=0)) #reverse one hot
    cls = classes[output_idx]
    plt.title("Prediction = " + str(cls) + " | Actual = " + str(classes[target_cpu[0].numpy()]) )
    data_cpu = data.to(torch.device("cpu"))
    img = unnormTensor(data_cpu[0], normalized)
    plt.imshow(img, cmap = 'gray') 
    