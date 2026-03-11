import numpy as np

class Node():
    def __init__(self, dim: tuple, lr=0.02,tau_v = 15, tau_trace = 100):
        self.layer_pre = Layer(dim[0],tau_v,tau_trace)
        self.layer_post = Layer(dim[1],tau_v,tau_trace)
        self.W = np.random.uniform(0,0.1,(dim[0],dim[1]))
        self.lr = lr

    def step(self,timeline):
        self.drop_param()
        for I in timeline:
            self.layer_pre.LIF(I)
            self.layer_post.LIF(self.layer_pre.Spike @ self.W)
            self.Potention()
            self.Depression()

    def Potention(self):
        if np.any(self.layer_post.Spike):
            idx_post = np.where(self.layer_post.Spike)[0]
            self.W[:, idx_post] += (1 - self.W[:, idx_post]) * self.layer_pre.Trace[:, np.newaxis]

    def Depression(self):
        if np.any(self.layer_pre.Spike):
            idx_pre = np.where(self.layer_pre.Spike)[0]
            self.W[idx_pre, :] -= self.W[idx_pre, :] * self.layer_post.Trace

    def drop_param(self):
        self.layer_pre.V.fill(0)
        self.layer_pre.Trace.fill(0)
        self.layer_post.V.fill(0)
        self.layer_post.Trace.fill(0)

class Layer():
    def __init__(self,n,tau_v,tau_trace):
        self.V = np.zeros(n)
        self.Trace = np.zeros(n)
        self.Spike = np.zeros(n)
        self.tau_v = tau_v
        self.tau_trace = tau_trace

    def LIF(self,I):
        self.Spike.fill(0)
        self.Trace -= self.Trace/self.tau_trace
        self.V += -self.V/self.tau_v + I
        self.Spike = self.V >= 1
        mask = self.Spike > 0
        self.Trace[mask] = 1
        self.V[mask] = 0