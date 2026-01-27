import numpy as np
import time
from layers import Layer
from trainer import Train
from dataloader import decode_idx1_ubyte, decode_idx3_ubyte

def normilize(images):
    normalize_images = np.empty_like(images)
    for i in range(len(images)):
        normalize_image = np.array(images[i]/255).astype(np.float32)
        normalize_images[i] = normalize_image
    return normalize_images

images = decode_idx3_ubyte(r"src/datasets/train-images.idx3-ubyte")
train_inputs = normilize(images)[0]
train_labels = decode_idx1_ubyte(r"src/datasets/train-labels.idx1-ubyte")[0]

# val_inputs = np.array(x[10:13])
# val_labels = np.array(y[10:13])

lr = 0.001
epochs = 30

start = time.time()
layer1 = Layer(784,1)
model = np.array([layer1])
trainer = Train(model)
trainer.Fit([train_inputs],[train_labels],lr,epochs,drop=True)
#trainer.valid(val_inputs,val_labels)
print(f"{time.time() - start:.10f}")