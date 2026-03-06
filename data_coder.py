from PIL import Image
import numpy as np


def timing_coder(image_path):
    image = Image.open(image_path)
    image_array = np.array(image).flatten()
    timelines_array = np.zeros((image_array.shape[0], int(256/1)))
    for i, value in enumerate (image_array):
        if value <=-1:
            pass
        else:
            position = 255 - value
            timelines_array[i][position] = 1
    return timelines_array