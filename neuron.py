import numpy as np

class Node():
    def __init__(self, dim: tuple, lr=0.1,tau_v = 10, tau_trace = 10,):
        self.layer_pre = Layer(dim[0],tau_v,tau_trace)
        self.layer_post = Layer(dim[1],tau_v,tau_trace)
        self.W = np.random.normal(0.5,0.3,(dim[0],dim[1]))
        self.E = np.zeros((dim[0],dim[1]))

        self.lr = lr

    def step(self,timeline):
        self.drop_param()
        for I in timeline:
            self.layer_pre.LIF(I)
            self.layer_post.LIF(self.layer_pre.Spike @ self.W)
            self.STDP()

    def STDP(self):
        if np.any(self.layer_pre.Spike):
            idx_pre = np.where(self.layer_pre.Spike)[0]
            self.W[idx_pre, :] -= self.lr * self.W[idx_pre, :] * self.layer_post.Trace
        if np.any(self.layer_post.Spike):
            idx_post = np.where(self.layer_post.Spike)[0]
            self.W[:, idx_post] += self.lr * (1 - self.W[:, idx_post]) * self.layer_pre.Trace[:, np.newaxis]

    def drop_param(self):
        self.layer_pre.V[:] = 0
        self.layer_pre.Trace[:] = 0
        self.layer_post.V[:] = 0
        self.layer_post.Trace[:] = 0

class Layer():
    def __init__(self,n,tau_v,tau_trace,Tresholds=False):
        self.V = np.zeros(n)
        self.Trace = np.zeros(n)
        self.Spike = np.zeros(n)
        self.Thresholds = np.ones(n) if Tresholds else None

        self.tau_v = tau_v
        self.tau_trace = tau_trace

    def LIF(self,I):
        self.Spike = self.Spike * 0
        self.Trace = self.Trace - self.Trace/self.tau_trace
        self.V = self.V - self.V/self.tau_v + I
        self.Spike = (self.V >= self.Thresholds).astype(float) if self.Thresholds is not None else (self.V >= 1.0).astype(float)
        mask = self.Spike > 0
        self.Trace[mask] = 1.0
        self.V[mask] = 0
        # if self.Thresholds is not None:
        #     self.Thresholds[mask] += 0.4
        #     self.Thresholds = np.maximum(1,self.Thresholds - 0.1)