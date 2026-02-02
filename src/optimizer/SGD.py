import numpy as np
from src import Tensor

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