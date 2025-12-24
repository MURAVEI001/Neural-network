import numpy as np
import pandas as pd
import time

def initialize_weight(unit_in,unit_out):
    np.random.seed(1)
    return np.random.normal(0, np.sqrt(2.0 / unit_in),(unit_in,unit_out))

def load_dataset():
    df = pd.read_csv('mnist_train.csv',header=None)
    labels_train = df.iloc[:,0].values
    images_train = df.iloc[:,1:].values
    images_train = images_train.astype("float32") / 255
    return images_train,labels_train

def w_sum(inputs,weight_matrix):
    return np.dot(inputs,weight_matrix) # скалярное умножение inputs(n,) @ weight_matrix(len(inputs),m) = pred(m,)

def loss_function(pred,label):
    error = np.empty(len(pred))
    error = (pred - label) ** 2
    return error

def optimizer(inputs,pred,label,weight_matrix):
    weight_matrix -= np.dot(inputs,(pred - label)) * alpha # внешнее произведение pred(n,)-label(n,) @ inputs(m,) = weight_matrix(n,m)
    return weight_matrix

def relu(inputs):
    return [max(0,i) for i in inputs]

def relu2deriv(inputs):
    return [min(1,i) for i in inputs]

def neural_network(inputs,weight_matrix):
    pred = w_sum(inputs,weight_matrix)
    return pred

start = time.time()
alpha = 0.001
unit_in = 784
unit_out = 1
weight_matrix = (initialize_weight(unit_in,unit_out))
x,y = load_dataset()
batch_x = x[:100]
batch_y = y[:100]
batch_x_val = x[-10:]
batch_y_val = y[-10:]
print(weight_matrix.shape)

for j in range(100):
    all_error_predict = 0
    for i in range(len(batch_x)):
        pred = neural_network(batch_x[i],weight_matrix)
        error = loss_function(pred,batch_y[i])
        all_error_predict += error
        weight_matrix = optimizer(batch_x[i],pred,batch_y[i],weight_matrix)
        print(f"Pred: {pred} | label: {batch_y[i]}")
    print(f"Epochs: {j+1} | Error: {all_error_predict}")
all_error_val = 0
for j in range(len(batch_x_val)):
    valid_pred = w_sum(batch_x_val[j],weight_matrix)
    valid_error = loss_function(valid_pred,batch_y_val[j])
    all_error_val += valid_error
    print(f"Validation | Pred: {valid_pred} | Label: {batch_y_val[j]} | Error: {valid_error}")
print(f"Error: {all_error_val/len(batch_y_val)}")
print(f"{time.time() - start:.10f}") 

# 0.44 для 100x 10y 100ep
# 5.39 для 1000x 10y 100ep 

# // Architecture //
# 1. Layer | Input  | (784)
# 2. Layer | Output | (1)