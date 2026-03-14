import numpy as np
import torch

class Node():
    def __init__(self, dim: tuple, lr=0.01, tau_v = 60, tau_trace = 60, A_plus=0.12, A_minus=0.01,Help_k=0.005,device='cpu'):
        self.device = device
        self.layer_pre = Layer(dim[0],device=self.device)
        self.layer_post = Layer(dim[1],device=self.device)
        self.W = torch.empty((dim[0],dim[1]),device=self.device).uniform_(0,0.1)
        self.E = torch.zeros_like(self.W,device=self.device)
        self.lr = lr
        self.k = torch.zeros((dim[1]),device=self.device)
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.Help_k = Help_k
        torch.set_printoptions(threshold=torch.inf)

    def Fit(self,batch,label):
        self.drop_param()
        self.layer_pre.FPT(batch)
        self.layer_post.FPT(torch.matmul(self.layer_pre.s_prev,self.W))
        self.STDP()
        print(self.layer_post.s_prev[:,0])
        print(self.layer_post.s_prev[:,1])
        # self.k += self.layer_post.s_prev
        # self.R_STDP(label)

    def STDP(self):
        # Потенциация: post_spikes[t,b,k] * pre_traces[t,b,j] -> сумма по t,b -> матрица (N_pre, N_post)
        # Используем einsum: 'tbk,tbj->jk' (j - пре, k - пост)
        delta_plus = torch.einsum('tbk,tbj->jk', self.layer_post.s_prev, self.layer_pre.trace)  # форма (N_pre, N_post)

        # Депрессия: pre_spikes[t,b,j] * post_traces[t,b,k] -> сумма по t,b -> матрица (N_pre, N_post)
        delta_minus = torch.einsum('tbj,tbk->jk', self.layer_pre.s_prev, self.layer_post.trace)

        # Общее изменение
        self.W += self.A_plus * delta_plus - self.A_minus * delta_minus

    def drop_param(self):
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
    def __init__(self,n,device):
        self.device = device
        self.tau_trace = 100
        self.V_th = 1
        self.tau_v = 100
        self.K = 3

    def FPT(self,I):
        T = I.shape[0]
        device = self.device
        batch_shape = I.shape[1:]
        
        # Начальный потенциал
        u0 = torch.zeros(batch_shape, device=device)
        
        # Инициализация
        self.u_prev = torch.zeros(T, *batch_shape, device=device)
        self.s_prev = torch.zeros(T, *batch_shape, device=device)
        
        # Инициализация трасс (экспоненциальное затухание)
        self.trace = torch.zeros(T, *batch_shape, device=device)
        
        for _ in range(self.K):
            # Сдвиг для учёта предыдущего шага
            u_prev_shifted = torch.cat([u0.unsqueeze(0), self.u_prev[:-1]], dim=0)
            s_prev_shifted = torch.cat([torch.zeros_like(u0.unsqueeze(0)), 
                                       self.s_prev[:-1]], dim=0)
            

            u_new = (u_prev_shifted - self.V_th * s_prev_shifted)/self.tau_v + I

            # Генерация спайков
            s_new = (u_new >= self.V_th).float()
            
            # Обновление трасс (экспоненциальное затухание + прибавление при спайке)
            trace_new = self.trace * torch.exp(-torch.ones_like(self.trace) / self.tau_trace)
            trace_new = trace_new + s_new  # добавляем 1 при спайке
            
            # Сохраняем для следующей итерации
            self.u_prev, self.s_prev, self.trace = u_new, s_new, trace_new