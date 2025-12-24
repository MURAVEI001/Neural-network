import numpy as np
import pandas as pd
import time

class DataLoader:
    def __init__(self):
        pass

    def dataset_mnist(self):
        df = pd.read_csv('mnist_train.csv',header=None)
        self.y = df.iloc[:,0].values
        self.x = (df.iloc[:,1:].values).astype("float32") /255
        return self.x, self.y

class LossFunction:
    def __init__(self):
        pass

    def MSE(self,pred,label):
        self.error = np.empty(len(pred))
        self.error = (pred - label) ** 2
        return self.error

class Optimizer:
    def __init__(self,alpha):
        self.alpha = alpha

    def SGD(self,layers,label):
        #layer.w_matrix -= np.outer(layer.data_in,(layer.pred - label)) * self.alpha
        for i in range(len(layers)):
            layers[i].w_matrix = 

class Activaion:
    def ReLu(inputs):
        return [max(0,i) for i in inputs]
    def ReLu2deriv(inputs):
        return [min(1,i) for i in inputs]
    
class Layer:
    def __init__(self,unit_in,unit_out):
        self.unit_in = None
        self.unit_out = None
        self.data_in = None
        self.pred = None
        super().__init__()
        self.unit_in = unit_in
        self.unit_out = unit_out
        self.initialize_weight()
    
    def initialize_weight(self):
        np.random.seed(1)
        self.w_matrix = np.random.normal(0, np.sqrt(2.0 / self.unit_in),(self.unit_in,self.unit_out))
    
    def w_sum(self,data_in):
        self.data_in = data_in
        self.pred = np.dot(self.data_in,self.w_matrix)
        return self.pred

alpha = 0.01
epochs = 100

start = time.time()

data = DataLoader()
x,y = data.dataset_mnist()

batch_x = x[:100]
batch_y = y[:100]
batch_x_val = x[-100:]
batch_y_val = y[-100:]

layer1 = Layer(784,196)
layer2 = Layer(196,1) 


for i in range(epochs):
    all_loss = 0
    for j in range(len(batch_x)):
        pred1 = Activaion.ReLu(layer1.w_sum(batch_x[j]))
        pred2 = layer2.w_sum(pred1)
        loss = LossFunction()
        error = loss.MSE(pred2,y[j])
        all_loss += error
        optimizer = Optimizer(alpha)
        optimizer.SGD([layer2,layer1],y[j])
        print(f"Pred: {pred2} | Label: {batch_y[j]} |Error: {error}")
    print(f"Epoch: {i+1} |Error: {all_loss/len(batch_x)}")
all_loss_val = 0
for j in range(len(batch_x_val)):
    valid_pred1 = layer1.w_sum(batch_x_val[j])
    valid_pred2 = layer2.w_sum(valid_pred1)
    valid_error = loss.MSE(valid_pred2,batch_y_val[j])
    all_loss_val += valid_error
    print(f"Valid | Predict: {valid_pred2} | Label: {batch_y_val[j]} | Error: {valid_error}")
print(f"Error: {all_loss_val/len(batch_y)}")
print(f"{time.time() - start:.10f}") 

# 0.63 100x 10y 100ep
# 5.81 1000x 10y 100ep