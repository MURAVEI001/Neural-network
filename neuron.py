import numpy as np

class Neuron():
    def __init__(self,n_out=0):
        self.V = 0
        self.spike = 0
        self.trace = 0
        self.out = np.zeros(n_out)
        self.W = np.random.random(size=n_out)
        self.v_list = []
        self.w_list = []

    def LIF(self,I):
        self.spike = 0
        self.trace = self.trace - self.trace/20
        self.V = self.V - self.V/20 + I
        if self.V >= 1:
            self.spike = 1
            self.trace = 1
            self.V = 0
        self.out = self.spike * self.W
        self.v_list.append(self.V)

    def STDP(self,N_postList):
        lr = 0.1
        for i, n in enumerate(N_postList):
            if self.spike:
                self.q = self.W[i]
                self.W[i] = self.W[i] - lr * self.W[i] * n.trace
            if n.spike:
                self.W[i] = self.W[i] + lr * (1 - self.W[i]) * self.trace
        self.w_list.append(self.W)