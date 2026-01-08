import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataloader import decode_idx1_ubyte, decode_idx3_ubyte

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
    def tensor(self):
        return f" Data: {self.data} \n  Grad: {self.grad} \n Shape: {self.data.shape} \n Op: {self._op} \n Parents: {self._parents} \n Calc_grad: {self.calc_grad} \n Grad_calls: {self.fl_back}"

def add(a,b):
    c = Tensor(a.data + b.data, calc_grad=a.calc_grad or b.calc_grad, _op="add", _parents=(a,b))

    def _backward():
        if a.calc_grad:
            # dc/da = c.grad * 1
            grad = c.grad * 1
            a.accumulate_grad(grad)
            a.fl_back += 1 
        
        if b.calc_grad:
            # dc/db = c.grad * 1
            grad = c.grad * 1
            b.accumulate_grad(grad)
            b.fl_back += 1
    
    c._backward = _backward
    return c

def sub(a,b):
    b.data = b.data * -1
    return add(a,b)

def pow(a,exp):
    c = Tensor(a.data ** exp, calc_grad=a.calc_grad, _op="pow", _parents=(a,))

    def _backward():
        if a.calc_grad:
            # dc/da = c.grad * (exp*a.data**(exp-1))
            grad = c.grad * (exp * a.data**(exp-1))
            a.accumulate_grad(grad)
            a.fl_back += 1 
    
    c._backward = _backward
    return c

def mul(a,b):
    c = Tensor(a.data * b.data, calc_grad=a.calc_grad or b.calc_grad, _op="mul", _parents=(a,b))
    
    def _backward():
        if a.calc_grad:
            # dc/da = c.grad * b
            grad = c.grad * b.data
            a.accumulate_grad(grad)
            a.fl_back += 1 
        
        if b.calc_grad:
            # dc/db = c.grad * a
            grad = c.grad * a.data
            b.accumulate_grad(grad)
            b.fl_back += 1
    
    c._backward = _backward
    return c

def MSE(predict,target):
    p1 = sub(predict,target)
    p2 = pow(p1,2)
    loss = Tensor(p2.data,calc_grad=predict.calc_grad or target.calc_grad, _op="MSE", _parents=(predict,target))

    def _backward():
        if predict.calc_grad:
            # dc/da = c.grad * 2*(predict-target)
            grad = loss.grad * 2*(predict.data - target.data)
            predict.accumulate_grad(grad)
            predict.fl_back += 1 
        
        if target.calc_grad:
            # dc/db = c.grad * 2*(predict - targer)
            grad = loss.grad * 2*(predict.data - target.data)
            target.accumulate_grad(grad)
            target.fl_back += 1
    
    loss._backward = _backward
    return loss


def matmul(a,b):
    # a (m,n) @ b (n,p) = c (m,p)
    c = Tensor(a.data @ b.data, calc_grad=a.calc_grad or b.calc_grad, _op="matmul", _parents=(a,b))
    
    def _backward():
        if a.calc_grad:
            # dc/da = c.grad @ b = (m,p) @ (n,p).T 
            grad = c.grad @ T(b).data
            a.accumulate_grad(grad)
            a.fl_back += 1

        if b.calc_grad:
            # dc/db = c.grad @ a = (m,n).T @ (m,p)
            grad = T(a).data @ c.grad
            b.accumulate_grad(grad)
            b.fl_back += 1
    
    c._backward = _backward
    return c

def T(a):
    # a (m,n) -> a (n,m).T
    c = Tensor(a.data.T, calc_grad=a.calc_grad, _op="transpone", _parents=(a,))

    def _backward():
        if a.calc_grad:
            # dc/da = c.grad.T
            grad = c.grad.T
            a.accumulate_grad(grad)
            a.fl_back += 1 
    
    c._backward = _backward
    return c

def backward(self):
    visited = set()
    topo_map = []

    def build_topo_map(self):
        if self not in visited:
            visited.add(self)
            for v in self._parents:
                build_topo_map(v)
            topo_map.append(self)
                        
    build_topo_map(self)

    for node in reversed(topo_map):
        if self == node:
            self.grad = np.ones_like(self.data)
            node._backward()
        else:
            node._backward()

def build_graph(G,self):
    G.add_node(f"{self.data.shape} {self._op} {self.fl_back}")
    for p in self._parents:
        G.add_node(f"{p.data.shape} {p._op} {p.fl_back}")
        G.add_edge(f"{p.data.shape} {p._op} {p.fl_back}",f"{self.data.shape} {self._op} {self.fl_back}")
        G = build_graph(G,p)
    return G

images = decode_idx3_ubyte(r"src/datasets/train-images.idx3-ubyte")
labels = decode_idx1_ubyte(r"src/datasets/train-labels.idx1-ubyte")

np.random.seed(1)
x1 = Tensor([images[0]])
w1 = Tensor(np.random.normal(0, np.sqrt(2.0 / 784),(784, 10)),calc_grad=True)
x2 = matmul(x1,w1)
w2 = Tensor(np.random.normal(0, np.sqrt(2.0 / 10),(10, 1)),calc_grad=True) 
predict = matmul(x2,w2)
target = Tensor(labels[0])
loss = MSE(predict,target)
backward(loss)

G = nx.DiGraph()
G = build_graph(G,loss)
nx.draw(G, pos = nx.spring_layout(G),
        with_labels=True,
        arrows=True,
        arrowsize=20,
        arrowstyle='->') 
plt.show()