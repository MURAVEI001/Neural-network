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
        out = Tensor(other.data*-1, calc_grad=other.calc_grad, _op="sub", _parents=(other,))
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

def SGD(self,lr,parameters):
    visited = set()
    topo_map = []

    def build_topo_map(self):
        if self not in visited:
            visited.add(self)
            for v in self._parents:
                build_topo_map(v)
            if self.calc_grad:
                topo_map.append(self)

    def update_weight(parameters, lr):
        for param in parameters:
            param.data = param.data - lr * param.grad

    def zero_grad(topo_map):
        for i in topo_map:
            i.zeros_grad
                    
    build_topo_map(self)

    for node in reversed(topo_map):
            node._backward()
            node.fl_back += 1
    
    update_weight(parameters,lr)
    zero_grad(topo_map)

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
            loss = MSE(self.predict,Tensor(self.labels))
            SGD(loss,lr,self.learn_param)
            print(loss.data, self.predict.data)
        #view_graph(loss,graph="param")

class Layer:
    def __init__(self):
        self.prev_layer = None
        self.next_layer = None
        self.first_layer = False
        self.last_layer = False

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

images = decode_idx3_ubyte(r"src/datasets/train-images.idx3-ubyte",normilize=True)
labels = decode_idx1_ubyte(r"src/datasets/train-labels.idx1-ubyte")

start = time.time()
model = Model()
model.build(
    Dense(784,1)
)
model.train(images[0],labels[0],30)
print(f"{time.time() - start:.10f}") 
