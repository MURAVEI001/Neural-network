import numpy as np
import time
from layers import Layer
from trainer import Train
from dataload import dataset_mnist

x,y = dataset_mnist()
inputs = np.array(x[:500])
labels = np.array(y[:500])
lr = 0.01
epochs = 1

start = time.time()
layer1 = Layer(784,1)
model = np.array([layer1])
trainer = Train(model)
trainer.Fit(inputs,labels,lr,epochs)
print(f"{time.time() - start:.10f}")