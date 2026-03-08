from PIL import Image
import numpy as np
import os
from natsort import natsorted

def timing_coder(directory,count=1,allTime=256,power=1):
    images_dir = fr"{directory}\images"
    labels_dir = fr"{directory}\labels"

    imagesList = [os.path.join(images_dir, f) for f in natsorted(os.listdir(images_dir))[:count]]
    labelsList = [os.path.join(labels_dir, f) for f in natsorted(os.listdir(labels_dir))[:count]]

    img_list = []

    for file in imagesList:
        img_list.append(np.array(Image.open(file)))
    images = np.array(img_list)

    N, H, W = images.shape
    num_pixels = H * W
    images_flat = images.reshape(N, num_pixels)
    
    timelines = np.zeros((N, allTime, num_pixels), dtype=np.float32)
    
    for i in range(N):
        for j in range(num_pixels):
            value = images_flat[i, j]
            pos = int(255 - value)
            timelines[i, pos, j] = power

    label_list = []

    for file in labelsList:
        with open(file, 'r', encoding='utf-8') as f:
            label_list.append(f.read())
    labels = np.array(label_list)
    return timelines, labels