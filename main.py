import numpy as np
import time

from src import Tensor,Layer,Dense,view_graph,getMnistImage,getMnistLabel,SGD,MSE_loss

np.random.seed(1)

class Model:
    def __init__(self):
        self.layers = []
        self.learn_param = []

    def build(self,*layers):
        for i in range(len(layers)):
            self.learn_param.append(layers[i].w)
            self.layers.append(layers[i])

            if i == 0 and i == len(layers)-1:
                layers[i].first_layer = True
                layers[i].last_layer = True
            elif i == 0:
                layers[i].next_layer = layers[i+1]
                layers[i].first_layer = True
            elif i == len(layers)-1:
                layers[i].prev_layer = layers[i-1]
                layers[i].last_layer = True
            else:
                layers[i].prev_layer = layers[i-1]
                layers[i].next_layer = layers[i+1]

        #\view_graph(self.layers,graph="layer")

    def train(self,images,labels,epoch,lr=0.001):
        self.images = images
        self.labels = labels
        self.epoch = epoch
        self.lr = lr

        for i in range(epoch):
            self.predict = self.layers[0].forward(Tensor([images]))
            loss = MSE_loss(self.predict,Tensor(self.labels))
            SGD(loss,lr,self.learn_param)
            print(loss.data, self.predict.data)
        #view_graph(loss,graph="param")

images = getMnistImage(r"src/datasets/train-images.idx3-ubyte",normilize=True)
labels = getMnistLabel(r"src/datasets/train-labels.idx1-ubyte")

start = time.time()
model = Model()
model.build(
    Dense(784,1)
)
model.train(images[0],labels[0],30)
print(f"{time.time() - start:.10f}") 
