import numpy as np
import struct
import sys

# decode idx3-ubyte files
def getMnistImage(fileDir,normilize=False):
    
    bin_data = open(fileDir, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)   
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, image_size))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(image_size)
        offset += struct.calcsize(fmt_image)

    def normilizer(images):
        normalize_images = np.empty_like(images)
        for i in range(len(images)):
            normalize_image = np.array(images[i]/255).astype(np.float32)
            normalize_images[i] = normalize_image
        return normalize_images

    if normilize:
        normilize_images = normilizer(images)
        return normilize_images
    return images

# decode idx1-ubyte files
def getMnistLabel(fileDir):

    bin_data = open(fileDir, 'rb').read()
    
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels