from PIL import Image
import numpy as np
import os
from natsort import natsorted

def timing_coder(imagesList,labelsList,allTime,power):

    img_list = []

    for file in imagesList:
        img_list.append(np.array(Image.open(file)))

    images = np.array(img_list)

    N, H, W = images.shape
    num_pixels = H * W
    images_flat = np.array(images.reshape(N, num_pixels),dtype=np.int32)
    
    timelines = np.zeros((N, allTime, num_pixels), dtype=np.float32)
    
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
    labels = np.array(label_list, dtype=np.int8)
    
    return timelines, labels

def getTimeLine(directory,Train=1,Valid=1,allTime=256,power=1):
    images_dir = fr"{directory}\images"
    labels_dir = fr"{directory}\labels"

    images = np.array([os.path.join(images_dir, f) for f in natsorted(os.listdir(images_dir))])
    labels = np.array([os.path.join(labels_dir, f) for f in natsorted(os.listdir(labels_dir))])

    imagesTrain, labelsTrain = timing_coder(images[:Train],labels[:Train],allTime=allTime,power=power)
    imagesValid, labelsValid = timing_coder(images[-Valid:],labels[-Valid:],allTime=allTime,power=power)
    
    return imagesTrain,labelsTrain,imagesValid,labelsValid