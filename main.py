import numpy as np
import matplotlib.pyplot as plt

step_time = 1.0 # {ms}
all_time = 1000.0 # {ms}

time_line = np.zeros_like((np.arange(0,all_time,step_time)))
time_line[300] = 5.0

class Neuron():
    def __init__(self):
        self.V = 0
        self.spike = 0
        self.trace = 0
        self.V_list = []
        self.Spike_list = []

    def LIF(self,I):
        self.V = self.V - (self.V/10.0) + I
        self.logV
        self.spike = 0
        if self.V >= 1.0:
            self.fire
            self.V -= 1.0
            self.trace += 1.0
        self.trace = self.trace - self.trace/10.0
        self.logSpike

    @property
    def fire(self):
        self.spike = 1

    @property
    def logV(self):
        self.V_list.append(self.V)

    @property
    def logSpike(self):
        self.Spike_list.append(self.spike)

class Sinapse():
    def __init__(self,pre_neuron,post_neuron):
        self.w = 0.5
        self.lr = 0.5
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.W_list = []

    def STDP(self,I):
        if self.pre_neuron.spike:
            self.w = self.w - self.lr * (self.w - 0.0) * self.post_neuron.trace   
        if self.post_neuron.spike:
            self.w = self.w + self.lr * (1.0 - self.w) * self.pre_neuron.trace
        self.logW
        return self.w * I
    
    @property
    def logW(self):
        self.W_list.append(self.w)

    def predictor(self,y,t):
        if y:
            predict = t
            self.predict_list.append(predict)
            print(predict)

    def Fit(self):
        self.predict_list = []
        for epoch in range(5):    
            for t,i in enumerate(time_line):
                self.pre_neuron.LIF(i)
                i = self.STDP(self.pre_neuron.spike)
                self.post_neuron.LIF(i)
                predict = self.predictor(self.post_neuron.spike,t)                

neuron1 = Neuron()
neuron2 = Neuron()
sinapse1_2 = Sinapse(neuron1,neuron2)

sinapse1_2.Fit()

fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(8,4))

axes[0,0].plot(neuron1.V_list)
axes[0,0].set_title("V1")

axes[0,1].plot(neuron2.V_list)
axes[0,1].set_title("V2")

axes[1,0].plot(neuron1.Spike_list)
axes[1,0].set_title("Spike1")

axes[1,1].plot(neuron2.Spike_list)
axes[1,1].set_title("Spike2")

axes[2,0].plot(sinapse1_2.W_list)
axes[2,0].set_title("W")

axes[2,1].plot(sinapse1_2.predict_list)
axes[2,1].set_title("predict")
plt.show()