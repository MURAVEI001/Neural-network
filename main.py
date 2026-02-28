import numpy as np
import matplotlib.pyplot as plt

step_time = 1.0 # {ms}
all_time = 100.0 # {ms}

img = np.array([150],dtype=np.float16)/255
I_max = 2.0

time_line = np.tile(I_max*img,len(np.arange(0,all_time,step_time)))

class Neuron():
    def __init__(self):
        self.V = 0
        self.spike = 0
        self.t_spike = None
        self.trace = 0
        self.V_list = []
        self.Spike_list = []

    def LIF(self,I,t):
        self.V += (-self.V + I)/10.0
        self.logV
        self.spike = 0
        if self.V >= 1.0:
            self.makeSpike(t)
            self.V -= 1.0
            self.trace += 1.0
        self.logSpike 

    def makeSpike(self,t):
        self.spike = 1
        self.t_spike = t

    @property
    def logV(self):
        self.V_list.append(self.V)

    @property
    def logSpike(self):
        self.Spike_list.append(self.spike)

class Sinapse():
    def __init__(self,pre_neuron,post_neuron):
        self.w = 0.5
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.W_list = []

    def STDP(self):
        if self.post_neuron.t_spike != None and self.pre_neuron.t_spike != None:
            self.delta_t = self.post_neuron.t_spike - self.pre_neuron.t_spike
            if self.delta_t > 0:
                self.w += ((1.0 - self.w) * 0.1) * np.exp(-self.delta_t/10.0)
            else:
                self.w += ((self.w + 1.0) * 0.1) * np.exp(self.delta_t/10.0)
        print(self.w)
        self.logW
    
    @property
    def logW(self):
        self.W_list.append(self.w)

    def Fit(self):    
        for t,i in enumerate(time_line):
            self.pre_neuron.LIF(i,t)
            self.post_neuron.LIF(i,t)
            self.STDP()

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
plt.show()