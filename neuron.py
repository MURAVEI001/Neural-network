import numpy as np

class Node():
    def __init__(self, dim: tuple, lr=0.03,tau_v = 10, tau_trace = 10, tau_e = 10):
        self.layer_pre = Layer(dim[0],tau_v,tau_trace)
        self.layer_post = Layer(dim[1],tau_v,tau_trace,Tresholds=True)
        self.W = np.random.uniform(0,0.5,(dim[0],dim[1]))
        self.E = np.zeros_like(self.W)
        self.k = np.zeros(dim[1])
        self.lr = lr
        self.tau_e = tau_e

    def step(self,timeline,label):
        self.drop_param()
        for I in timeline:
            self.layer_pre.LIF(I)
            self.layer_post.LIF(self.layer_pre.Spike @ self.W)
            self.k += self.layer_post.Spike
            self.LateralInhibition()
            self.Potention()
            self.Depression()
            self.E *= np.exp(-1/self.tau_e)
        self.R_STDP(label)

    def R_STDP(self,label):
        if np.sum(self.k) == 0:
            reward = 0
        else:
            winner = np.argmax(self.k)
            reward = 1.0 if winner == label else -1.0

        self.W += self.lr * reward * self.E
        self.E.fill(0)
        self.k.fill(0)

    def Potention(self):
        if np.any(self.layer_post.Spike):
            idx_post = np.where(self.layer_post.Spike)[0]
            self.E[:, idx_post] += (1 - self.W[:, idx_post]) * self.layer_pre.Trace[:, np.newaxis]

    def Depression(self):
        if np.any(self.layer_pre.Spike):
            idx_pre = np.where(self.layer_pre.Spike)[0]
            self.E[idx_pre, :] -= self.W[idx_pre, :] * self.layer_post.Trace

    def LateralInhibition(self):
        spiked = np.where(self.layer_post.Spike > 0)[0]
        if len(spiked) > 1:
            winner = spiked[np.argmax(self.layer_post.V[spiked])]
            for j in spiked:
                if j != winner:
                    self.layer_post.Spike[j] = 0
                    self.layer_post.V[j] = 0

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
        self.Spike.fill(0)
        self.Trace -= self.Trace/self.tau_trace
        self.V += -self.V/self.tau_v + I
        self.Spike = (self.V >= self.Thresholds).astype(float) if self.Thresholds is not None else (self.V >= 1.0).astype(float)
        mask = self.Spike > 0
        self.Trace[mask] = 1.0
        self.V[mask] = 0
        if self.Thresholds is not None:
            self.Thresholds[mask] += 0.05
            self.Thresholds = np.maximum(1,self.Thresholds - 0.01)