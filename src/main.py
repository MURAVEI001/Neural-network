import numpy as np
from dataloader import decode_idx3_ubyte, decode_idx1_ubyte
import networkx as nx
import matplotlib.pyplot as plt

class Tensor:
    def __init__(self,data,requires_grad=False,_op=None,_parents=None):
        self.requires_grad=requires_grad
        self._op = _op
        self._parents = _parents if _parents is not None else ()
        self._backward = lambda: None
        self.grad = None
        self.backward_calls = 0

        if not isinstance(data,np.ndarray):
            data = np.array(data, dtype = np.float32)
        self.data = data.astype(np.float32)

        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def to_tensor(self,other):
        if not isinstance(other,Tensor):
            return Tensor(other)
        else:
            return other
    
    def accumulate_grad(self,grad):
        self.grad += grad

    @property
    def tensor(self):
        return f" Data: {self.data} \n Requires_grad: {self.requires_grad} \n op: {self._op} \n _parents: {self._parents} \n _backward: {self._backward} \n grad: {self.grad} \n calls: {self.backward_calls} \n"

    def __add__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data + other.data,requires_grad=self.requires_grad or other.requires_grad, _op="add", _parents=(self,other))
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * 1
                print(out.data)
                self.accumulate_grad(grad)
            
            if other.requires_grad:
                grad = out.grad * 1
                other.accumulate_grad(grad)

        out._backward = _backward
        return out
    
    def __sub__(self,other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __mul__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data * other.data,requires_grad=self.requires_grad or other.requires_grad, _op="mul", _parents=(self,other))

        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                self.accumulate_grad(grad)
            
            if other.requires_grad:
                grad = out.grad * self.data
                other.accumulate_grad(grad)
            
        out._backward = _backward
        return out

    def __matmul__(self,other):
        other = self.to_tensor(other)
        out = Tensor(self.data @ other.data,requires_grad=self.requires_grad or other.requires_grad, _op="matmul", _parents=(self,other))

        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.T
                self.accumulate_grad(grad)
            
            if other.requires_grad:
                grad = self.data.T @ out.grad
                other.accumulate_grad(grad)
            
        out._backward = _backward
        return out

    def __pow__(self,exponenta):
        out = Tensor(self.data ** exponenta,requires_grad=self.requires_grad, _op="pow", _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (exponenta*self.data**(exponenta-1))
                print(grad)
                self.accumulate_grad(grad)
            
        out._backward = _backward
        return out
    
    @property
    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _op="transpone", _parents=(self,))

        def _backward():
            if self.requires_grad:
                grad = out.grad.T
                self.accumulate_grad(grad)
            
        out._backward = _backward
        return out

class Layer:
    def __init__(self,dim_in,dim_out,activation=None):
        self.dim_in = dim_in
        self.dim_out = dim_out
        np.random.seed(1)
        self.w = Tensor(np.random.normal(0, np.sqrt(2.0 / self.dim_in),(self.dim_in, self.dim_out)),requires_grad=True)   

    @property
    def info(self):
        return f"W: {self.w.tensor} \n X: {self.x.tensor}"
        
    def forward(self,x):
        self.x = x
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


#images = decode_idx3_ubyte(r"src/datasets/train-images.idx3-ubyte")
#labels = decode_idx1_ubyte(r"src/datasets/train-labels.idx1-ubyte")

X = Tensor([1,2,3])

model = Seq(
    Layer(3,1),

    )

predict = model(X)
loss = MSE(predict,5)

# for i, layer in enumerate(model.layers):
#     print(i+1, layer.info)
# print(f"Output: {predict.tensor}")
# print(f"Loss: {loss.tensor}")

for i, layer in enumerate(model.layers):
    print(i+1, layer.info)
print(f"Output: {predict.tensor}")
print(f"Loss: {loss.tensor}")

def build_graph(G,self):
    G.add_node(f"{self.data.shape} {self._op}")
    for p in self._parents:
        G.add_node(f"{p.data.shape} {p._op}")
        G.add_edge(f"{p.data.shape} {p._op}",f"{self.data.shape} {self._op}")
        G = build_graph(G,p)
    return G

G = nx.DiGraph()
G = build_graph(G,loss)
nx.draw(G, pos = nx.spring_layout(G),
        with_labels=True,
        arrows=True,
        arrowsize=20,
        arrowstyle='->') 
plt.show()