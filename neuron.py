import numpy as np

class Node():
    def __init__(self,n_in,n_out):
        self.layer_pre = Layer(n_in)
        self.layer_post = Layer(n_out)
        self.W = np.random.normal(0.5,0.3,(n_in,n_out))

    def step(self,timeline):
        self.layer_pre.V[:] = 0
        self.layer_pre.Trace[:] = 0
        self.layer_post.V[:] = 0
        self.layer_post.Trace[:] = 0
        winner_found = False
        winner = -1
        for I in timeline:
            self.layer_pre.LIF(I)
            self.layer_post.LIF(self.layer_pre.Spike @ self.W)

            if not winner_found:
                spiked = np.where(self.layer_post.Spike)[0]
                if len(spiked) > 0:
                    winner = np.random.choice(spiked)
                    winner_found = True
                    for i in spiked:
                        if i != winner:
                            self.layer_post.Spike[i] = 0.0
                            self.layer_post.V[i] = 0.0
            else:
                mask = np.ones(len(self.layer_post.V), dtype=bool)
                mask[winner] = False
                self.layer_post.Spike[mask] = 0.0
                self.layer_post.V[mask] = 0.0
            self.STDP()

    def STDP(self):
        self.lr = 0.1
        if np.any(self.layer_pre.Spike):
            idx_pre = np.where(self.layer_pre.Spike)[0]
            self.W[idx_pre, :] -= self.lr * self.W[idx_pre, :] * self.layer_post.Trace
        if np.any(self.layer_post.Spike):
            idx_post = np.where(self.layer_post.Spike)[0]
            self.W[:, idx_post] += self.lr * (1 - self.W[:, idx_post]) * self.layer_pre.Trace[:, np.newaxis]     

class Layer():
    def __init__(self,n,Tresholds=False):
        self.V = np.zeros(n)
        self.Trace = np.zeros(n)
        self.Spike = np.zeros(n)
        self.Thresholds = np.ones(n) if Tresholds else None

    def LIF(self,I):
        self.Spike = self.Spike * 0
        self.Trace = self.Trace - self.Trace/20
        self.V = self.V - self.V/20 + I
        self.Spike = (self.V >= self.Thresholds).astype(float) if self.Thresholds is not None else (self.V >= 1.0).astype(float)
        mask = self.Spike > 0
        self.Trace[mask] = 1.0
        self.V[mask] = 0
        if self.Thresholds is not None:
            self.Thresholds[mask] += 0.4
            self.Thresholds = np.maximum(1,self.Thresholds - 0.1)