import numpy as np

class Node():
    def __init__(self, dim: tuple, lr=0.001,tau_v = 40, tau_trace = 40):
        self.layer_pre = Layer(dim[0],tau_v,tau_trace)
        self.layer_post = Layer(dim[1],tau_v,tau_trace)
        self.W = np.random.uniform(0,0.5,(dim[0],dim[1]))
        self.E = np.zeros_like(self.W)
        self.lr = lr
        self.k = np.zeros((dim[1]))
        #np.set_printoptions(threshold=np.inf)

    def step(self,timeline,label):
        self.drop_param()
        for I in timeline:
            self.layer_pre.LIF(I)
            self.layer_post.LIF(self.layer_pre.Spike @ self.W)
            self.Potention()
            self.Depression()
            self.k += self.layer_post.Spike
        self.R_STDP(label)

    def Potention(self):
        if np.any(self.layer_post.Spike):
            idx_post = np.where(self.layer_post.Spike)[0]
            self.E[:, idx_post] += (1 - self.W[:, idx_post]+self.E[:, idx_post]) * self.layer_pre.Trace[:, np.newaxis]

    def Depression(self):
        if np.any(self.layer_pre.Spike):
            idx_pre = np.where(self.layer_pre.Spike)[0]
            self.E[idx_pre, :] -= (self.W[idx_pre, :]+self.E[idx_pre, :]) * self.layer_post.Trace

    def drop_param(self):
        self.layer_pre.V.fill(0)
        self.layer_pre.Trace.fill(0)
        self.layer_post.V.fill(0)
        self.layer_post.Trace.fill(0)
        self.k.fill(0)
        self.E.fill(0)
    
    def R_STDP(self,label):
        predict = [np.argmax(self.k)]
        if predict[0] == label:
            self.W[:,predict[0]] += self.lr * self.E[:,predict[0]]
        else:
            self.W[:,predict[0]] -= self.lr * self.E[:,predict[0]]

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