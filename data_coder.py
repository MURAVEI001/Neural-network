from PIL import Image
import numpy as np
import os
from natsort import natsorted
import torch

def timing_coder(imagesList,labelsList,allTime,power,device):

    img_list = []

    for file in imagesList:
        img_list.append(np.array(Image.open(file)))

    images = np.array(img_list)

    N, H, W = images.shape
    num_pixels = H * W
    images_flat = np.array(images.reshape(N, num_pixels),dtype=np.int32)
    
    timelines = torch.zeros((N, allTime, num_pixels), dtype=torch.float32,device=device)
    
    for i in range(N):
        for j in range(num_pixels):
            value = images_flat[i, j]
            if value < 30:
                pass
            else:
                pos = (allTime-1)-value
                timelines[i, pos, j] = power

    label_list = []

    for file in labelsList:
        with open(file, 'r', encoding='utf-8') as f:
            label_list.append(f.read())
    label_list = np.array(label_list, dtype=np.int8)
    labels = torch.tensor(label_list, dtype=torch.int8)
    
    return timelines, labels

def getTimeLine(directory,Train=1,Valid=1,allTime=256,power=1,device='cpu'):
    images_dir = fr"{directory}\images"
    labels_dir = fr"{directory}\labels"

    images = np.array([os.path.join(images_dir, f) for f in natsorted(os.listdir(images_dir))])
    labels = np.array([os.path.join(labels_dir, f) for f in natsorted(os.listdir(labels_dir))])

    imagesTrain, labelsTrain = timing_coder(images[:Train],labels[:Train],allTime=allTime,power=power,device=device)
    imagesValid, labelsValid = timing_coder(images[-Valid:],labels[-Valid:],allTime=allTime,power=power,device=device)
    
    return imagesTrain,labelsTrain,imagesValid,labelsValid