import numpy as np
from dataloader import decode_idx1_ubyte, decode_idx3_ubyte
from build_graph import view_graph
import time

np.random.seed(1)

class Tensor:
    def __init__(self, data, calc_grad=False, _op=None, _parents=None):
        self._op = _op
        self._parents = _parents if _parents is not None else ()
        self.calc_grad = calc_grad
        self._backward = lambda: None
        self.fl_back = 0

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(self.data)

    def to_tensor(other):
        if not isinstance(other, Tensor):
            return Tensor(other)
        else:
            return other
    
    def accumulate_grad(self, grad):
        self.grad += grad

    @property
    def zeros_grad(self):
        self.grad = np.zeros_like(self.grad)

    @property
    def tensor(self):
        return f" Data: {self.data} \n  Grad: {self.grad} \n Shape: {self.data.shape} \n Op: {self._op} \n Parents: {self._parents} \n Calc_grad: {self.calc_grad} \n Grad_calls: {self.fl_back}"

    @property
    def T(self):
        # a (m,n) -> a (n,m).T
        out = Tensor(self.data.T, calc_grad=self.calc_grad, _op="transpone", _parents=(self,))

        def _backward():
            if self.calc_grad:
                # dc/da = out.grad.T
                grad = out.grad.T
                self.accumulate_grad(grad)
        
        out._backward = _backward
        return out

    def __add__(self,other):
        # a + b = c   
        # a - self  
        # b - other
        # c - out
        out = Tensor(self.data + other.data, calc_grad=self.calc_grad or self.calc_grad, _op="add", _parents=(self,other))

        def _backward():
            if self.calc_grad:
                # dc/da = out.grad * 1
                grad = out.grad * 1
                self.accumulate_grad(grad)
            
            if other.calc_grad:
                # dc/db = out.grad * 1
                grad = out.grad * 1
                other.accumulate_grad(grad)
        
        out._backward = _backward
        return out

    def __sub__(self,other):
        out = Tensor(other.data*-1, calc_grad=other.calc_grad, _op="neg", _parents=(other,))
        return self+out

    def __pow__(self,exp):
        out = Tensor(self.data ** exp, calc_grad=self.calc_grad, _op="pow", _parents=(self,))

        def _backward():
            if self.calc_grad:
                # dc/da = out.grad * (exp*self.data**(exp-1))
                grad = out.grad * (exp * self.data**(exp-1))
                self.accumulate_grad(grad)
        
        out._backward = _backward
        return out

    def __mul__(self,other):
        out = Tensor(self.data * other.data, calc_grad=self.calc_grad or other.calc_grad, _op="mul", _parents=(self,other))
        
        def _backward():
            if self.calc_grad:
                # dc/da = out.grad * other.data
                grad = out.grad * other.data
                self.accumulate_grad(grad)
            
            if other.calc_grad:
                # dc/db = out.grad * self.data
                grad = out.grad * self.data
                other.accumulate_grad(grad)
        
        out._backward = _backward
        return out

    def __matmul__(self,other):
        # a (m,n) @ b (n,p) = c (m,p)
        out = Tensor(self.data @ other.data, calc_grad=self.calc_grad or other.calc_grad, _op="matmul", _parents=(self,other))
        
        def _backward():
            if self.calc_grad:
                # dc/da = out.grad @ other.data.T = (m,p) @ (n,p).T 
                grad = out.grad @ other.data.T
                self.accumulate_grad(grad)

            if other.calc_grad:
                # dc/db = c.grad @ self.data.T = (m,n).T @ (m,p)
                grad = self.data.T @ out.grad
                other.accumulate_grad(grad)
        
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(axis=None, keepdims=False), calc_grad=self.calc_grad, _op="mean", _parents=(self,))
        
        def _backward():
            if self.calc_grad:
                # dc/da = out.grad * 1/n  
                # n - размер batch
                grad = out.grad * 1/out.data.size
                self.accumulate_grad(grad)
        
        out._backward = _backward
        return out

def MSE(predict,target):
    # MSE - (predict - target)**2 / n
    loss = Tensor(((predict-target)**2).mean().data, calc_grad=predict.calc_grad or target.calc_grad, _op="MSE", _parents=(predict,target))
    loss.grad = np.ones_like(loss.data)

    def _backward():
        if predict.calc_grad:
            # dc/da = c.grad * 2*(predict-target)
            grad = loss.grad * 2*(predict.data - target.data)
            predict.accumulate_grad(grad)
        
        if target.calc_grad:
            # dc/db = c.grad * 2*(predict - targer)
            grad = loss.grad * 2*(predict.data - target.data)
            target.accumulate_grad(grad)
    
    loss._backward = _backward
    return loss

def backward(self):
    visited = set()
    topo_map = []

    def build_topo_map(self):
        if self not in visited:
            visited.add(self)
            for v in self._parents:
                build_topo_map(v)
            if self.calc_grad:
                topo_map.append(self)
                    
    build_topo_map(self)

    for node in reversed(topo_map):
            node._backward()
            node.fl_back += 1
    
    return topo_map

def update_weight(parameters, alpha=0.001):
    for self in parameters:
        if self.calc_grad:
            self.data = self.data - alpha * self.grad
        self.zeros_grad

def zeros_grad(topo_map):
    for self in topo_map:
        self.zeros_grad

images = decode_idx3_ubyte(r"src/datasets/train-images.idx3-ubyte",normilize=True)
labels = decode_idx1_ubyte(r"src/datasets/train-labels.idx1-ubyte")

parameters = []

start = time.time()
x1 = Tensor([images[0]])
target = Tensor(labels[0])
w1 = Tensor(np.random.normal(0, np.sqrt(2.0 / 784),(784, 1)),calc_grad=True)
parameters.append(w1)
epoch = 30
for i in range(epoch):
    predict = x1@w1
    loss = MSE(predict,target)
    print(f"Epoch: {i+1} Loss: {loss.data}, \n Predict: {predict.data} \n Target: {target.data} \n ")
    topo_map = backward(loss)
    update_weight(parameters)
    zeros_grad(topo_map)
print(f"{time.time() - start:.10f}") 
view_graph(loss)
# 0.0009
# 0.009