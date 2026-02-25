import numpy as np
import matplotlib.pyplot as plt

step_time = 1.0 # {ms}
all_time = 100000.0 # {ms}

time_line = np.zeros_like(np.arange(0,all_time,step_time),dtype=np.float32)
time_line[100:80000:20] = 1.67

def LIF1(V,V_thr,I,tau,step_time):
    V = np.exp(-step_time/tau) * V + I
    spike = 0
    if V >= V_thr:
        spike = 1
        return spike, V - V_thr
    return spike, V

def LIF2(V,V_thr,I,tau,step_time,w):
    V = np.exp(-step_time/tau) * V + w * I
    spike = 0
    if V >= V_thr:
        spike = 1
        return spike, V - V_thr
    return spike, V

def STDP(tau,w,t1,t2):
    a = 0.1
    if t2 != None and t1 != None:
        delta_t = t2-t1
        if delta_t >= 0:
            w += a * np.exp(-delta_t/tau)
        elif delta_t <0:
            w -= a * np.exp(delta_t/tau)
    return w

w = 1
V1_list = []
V2_list = []
W_list = []
spike1_list = []
spike2_list = []
V1 = 0.0
V2 = 0.0
V1_list.append(V1)
V2_list.append(V2)
W_list.append(w)
t1 = None
t2 = None

for t, i in enumerate(time_line):
    spike1, V1 = LIF1(V=V1,V_thr=1.0,I=i,tau=10.0,step_time=step_time)
    spike1_list.append(spike1)
    if spike1:
        t1 = t
    spike2, V2 = LIF2(V=V2,V_thr=1.0,I=spike1,tau=10.0,step_time=step_time,w=w)
    spike2_list.append(spike2)
    if spike2:
        t2 = t
    w = STDP(tau=10.0,w=w,t1=t1,t2=t2)

    V1_list.append(V1)
    V2_list.append(V2)
    W_list.append(w)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,8))
axes[0,0].plot(time_line)
axes[0,0].set_title("Time line")

axes[0,1].plot(W_list)
axes[0,1].set_title("Weight")

axes[1,0].plot(V1_list)
axes[1,0].set_title("V1")

axes[1,1].plot(V2_list)
axes[1,1].set_title("V2")

axes[2,0].plot(spike1_list)
axes[2,0].set_title("spike1")

axes[2,1].plot(spike2_list)
axes[2,1].set_title("spike2")
plt.show()