import numpy as np

class Layer:
    def __init__(self,unit_in,unit_out):
        self.unit_in = unit_in
        self.unit_out = unit_out
        self.output = None
        self.act_output = None
        self.w = None
        self.delta = None
        
        if self.w==None:
            self.w_initialize()
        print("w init")
        print(self.__str__())
    
    def w_initialize(self):
        np.random.seed(1)
        self.w = (np.random.normal(0, np.sqrt(2.0 / self.unit_in),(self.unit_in,self.unit_out))).astype(np.float64)

    def w_summ(self,data_in):
        self.data_in = data_in
        self.output = np.dot(self.data_in,self.w)
        return self.output
    
    def __str__(self):
        info = f"Слой input|{self.unit_in} ---> {self.unit_out}|output W: {self.w.shape}"
        return info