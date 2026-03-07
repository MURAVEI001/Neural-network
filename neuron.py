import numpy as np

class Neuron():
    def __init__(self,n_out=0):
        self.V = 0
        self.spike = 0
        self.trace = 0
        self.out = np.zeros(n_out)
        self.W = np.random.random(size=n_out)
        self.threshold = 1.0  # начальный порог
        self.theta_increment = 0.05  # на сколько растёт после спайка
        self.theta_decay = 0.001  # скорость возврата
        self.v_list = []

    def LIF(self,I):
        self.spike = 0
        self.trace = self.trace - self.trace/100
        self.V = self.V - self.V/100 + I
        if self.V >= self.threshold:
            self.spike = 1
            self.trace = 1
            self.V = 0
            self.threshold += self.theta_increment  # повышаем порог после спайка
            # медленно возвращаем порог к базовому значению
        self.threshold = max(1.0, self.threshold - self.theta_decay)
        self.out = self.spike * self.W
        self.v_list.append(self.V)

    def STDP(self,N_postList):
        lr = 0.01
        for i, n in enumerate(N_postList):
            if self.spike:
                self.q = self.W[i]
                self.W[i] = self.W[i] - lr * self.W[i] * n.trace
            if n.spike:
                self.W[i] = self.W[i] + lr * (1 - self.W[i]) * self.trace
