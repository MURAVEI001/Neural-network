import numpy as np

class Tensor:
    def __init__(self,data,requires_grad=False,_op=None,_parents=None):
        self.requires_grad=requires_grad
        self._op = _op
        self._parents = _parents if _parents is not None else ()
        self._backward = lambda: None

        if not isinstance(data,np.ndarray):
            data = np.array(data, dtype = np.float32)
        self.data = data.astype(np.float32)

    def to_tensor(self,other):
        if not isinstance(other,Tensor):
            return Tensor(other)
        else:
            return other
        
    def __add__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data + other.data,requires_grad=self.requires_grad or other.requires_grad, _op="add", _parents=(self,other))
        return out
    
    def __mul__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data * other.data,requires_grad=self.requires_grad or other.requires_grad, _op="mul", _parents=(self,other))
        return out

    def __matmul__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data @ other.data,requires_grad=self.requires_grad or other.requires_grad, _op="matmul", _parents=(self,other))
        return out

class Layer:
    def __init__(self,dim_in,dim_out,activation=None):
        self.dim_in = dim_in
        self.dim_out = dim_out
        np.random.seed(1)
        self.w = Tensor(np.random.normal(0, np.sqrt(2.0 / self.dim_in),(self.dim_in, self.dim_out)))   

    def forward(self,x):
        out = x @ self.w
        return out

class Seq:
    def __init__(self, *layer):
        self.layers = layer
    
    def forward(self,x):
        return x 

def dataset_mnist():
    import pandas as pd
    df = pd.read_csv('mnist_train.csv',header=None)
    y = df.iloc[:,0].values
    x = (df.iloc[:,1:].values).astype("float32") /255
    return x, y

x, y = dataset_mnist()
model = Seq(
    Layer(784,2),
    Layer(2,1)
    )