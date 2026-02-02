import numpy as np
from src import Tensor

def MSE_loss(predict,target):
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