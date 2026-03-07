from PIL import Image
import numpy as np

def timing_coder(image_path, time_steps=256,power=1):
    img_list = []
    for path in image_path:
        img = Image.open(path)
        img_list.append(np.array(img))
    images = np.array(img_list)

    N, H, W = images.shape
    num_pixels = H * W
    images_flat = images.reshape(N, num_pixels)
    
    timelines = np.zeros((N, time_steps, num_pixels), dtype=np.float32)
    
    for i in range(N):
        for j in range(num_pixels):
            value = images_flat[i, j]
            pos = int(255 - value)
            timelines[i, pos, j] = power

    return timelines