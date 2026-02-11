import numpy as np

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