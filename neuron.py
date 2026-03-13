import numpy as np
import torch

class Node():
    def __init__(self, dim: tuple, lr=0.01, tau_v = 60, tau_trace = 60, A_plus=0.22, A_minus=0.1,Help_k=0.005,device='cpu'):
        self.device = device
        self.layer_pre = Layer(dim[0],tau_v,tau_trace,device=self.device)
        self.layer_post = Layer(dim[1],tau_v,tau_trace,device=self.device)
        self.W = torch.empty((dim[0],dim[1]),device=self.device).uniform_(0,0.1)
        self.E = torch.zeros_like(self.W,device=self.device)
        self.lr = lr
        self.k = torch.zeros((dim[1]),device=self.device)
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.Help_k = Help_k

    def Fit(self,timeline,label):
        self.drop_param()
        for I in timeline:
            self.layer_pre.LIF(I)
            self.layer_post.LIF(self.layer_pre.Spike @ self.W)
            self.Potention()
            self.Depression()
            self.k += self.layer_post.Spike
        self.R_STDP(label)
    
    def Valid(self,timeline,label):
        self.drop_param()
        for I in timeline:
            self.layer_pre.LIF(I)
            self.layer_post.LIF(self.layer_pre.Spike @ self.W)
            self.k += self.layer_post.Spike
        print(f"Predict: {torch.argmax(self.k).item()} || {label} :Label")

    def Potention(self):
        idx_post = torch.where(self.layer_post.Spike)[0]
        self.E[:, idx_post] += self.A_plus * (1 - self.W[:, idx_post]+self.E[:, idx_post]) * self.layer_pre.Trace[:, torch.newaxis]

    def Depression(self):
        idx_pre = torch.where(self.layer_pre.Spike)[0]
        self.E[idx_pre, :] -= self.A_minus *(self.W[idx_pre, :]+self.E[idx_pre, :]) * self.layer_post.Trace

    def drop_param(self):
        self.layer_pre.V.fill_(0)
        self.layer_pre.Trace.fill_(0)
        self.layer_post.V.fill_(0)
        self.layer_post.Trace.fill_(0)
        self.k.fill_(0)
        self.E.fill_(0)
    
    def R_STDP(self,label):
        predict = torch.argmax(self.k).item()
        if predict == label:
            self.W[:,predict] += self.lr * self.E[:,predict]
        else:
            self.W[:,predict] -= self.lr * self.E[:,predict]
            self.W[:,label] += self.Help_k * self.E[:,label]

class Layer():
    def __init__(self,n,tau_v,tau_trace,device):
        self.V = torch.zeros(n,device=device)
        self.Trace = torch.zeros(n,device=device)
        self.Spike = torch.zeros(n,dtype=torch.float,device=device)
        self.tau_v = tau_v
        self.tau_trace = tau_trace

    def LIF(self,I):
        self.Trace -= self.Trace/self.tau_trace
        self.V += -self.V/self.tau_v + I
        self.Spike = (self.V>=1).detach().clone().to(torch.float)
        mask = self.Spike > 0
        self.Trace[mask] = 1
        self.V[mask] = 0