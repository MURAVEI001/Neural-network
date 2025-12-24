import numpy as np
import time
from layers import Layer
from trainer import Train
from dataload import dataset_mnist

x,y = dataset_mnist()
inputs = np.array(x[:1000])
labels = np.array(y[:1000])
lr = 0.001
epochs = 10

start = time.time()
layer1 = Layer(784,1)
#layer2 = Layer(350,1)
model = np.array([layer1])
trainer = Train(model)
trainer.Fit(inputs,labels,lr,epochs)
print(f"{time.time() - start:.10f}")