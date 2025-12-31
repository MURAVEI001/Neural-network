import numpy as np
import time
from layers import Layer
from trainer import Train
from dataload import dataset_mnist

x,y = dataset_mnist()
train_inputs = np.array(x[:10])
train_labels = np.array(y[:10])

val_inputs = np.array(x[10:13])
val_labels = np.array(y[10:13])

lr = 0.001
epochs = 100

start = time.time()
layer1 = Layer(784,1)
model = np.array([layer1])
trainer = Train(model)
trainer.Fit(train_inputs,train_labels,lr,epochs,drop=True)
trainer.valid(val_inputs,val_labels)
print(f"{time.time() - start:.10f}")