import numpy as np
from dataloader import decode_idx3_ubyte, decode_idx1_ubyte

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
    
    def __sub__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data - other.data,requires_grad=self.requires_grad or other.requires_grad, _op="sub", _parents=(self,other))
        return out

    def __mul__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data * other.data,requires_grad=self.requires_grad or other.requires_grad, _op="mul", _parents=(self,other))
        return out

    def __matmul__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data @ other.data,requires_grad=self.requires_grad or other.requires_grad, _op="matmul", _parents=(self,other))
        return out

    def __pow__(self,exponenta):
        out = Tensor(self.data ** exponenta,requires_grad=self.requires_grad, _op="pow", _parents=(self,))
        return out

class Layer:
    def __init__(self,dim_in,dim_out,activation=None):
        self.dim_in = dim_in
        self.dim_out = dim_out
        np.random.seed(1)
        self.w = Tensor(np.random.normal(0, np.sqrt(2.0 / self.dim_in),(self.dim_in, self.dim_out)))   

    def forward(self,x):
        self.x = Tensor(x)
        if self.x.data.shape:
            out = self.x @ self.w
            return out
        else:
            out = self.x * self.w
            return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Seq:
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def MSE(predict, target):
    return (predict-target)**2


images = decode_idx3_ubyte(r"src/datasets/train-images.idx3-ubyte")
labels = decode_idx1_ubyte(r"src/datasets/train-labels.idx1-ubyte")

model = Seq(
    Layer(784,1),
    )

predict = model(images[0])
loss = MSE(predict,labels[0])
print(loss.data,predict.data)