import numpy as np
from src.tn.tensor import Tensor
from src.layers.layer import Layer

class Dense(Layer):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = Tensor(np.random.normal(0, np.sqrt(2.0 / self.in_dim),(self.in_dim, self.out_dim)),calc_grad=True)

    @property
    def previos(self):
        print(type(self), self.in_dim,self.out_dim)
        if self.prev_layer != None:
            self.prev_layer.previos

    @property
    def next(self):
        print(type(self), self.in_dim,self.out_dim)
        if self.next_layer != None:
            self.next_layer.next

    def forward(self,x):
        y = x @ self.w
        if self.last_layer == False:
            return self.next_layer.forward(y)
        else:
            return y