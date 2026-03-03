import numpy as np
import matplotlib.pyplot as plt

step_time = 1.0 # {ms}
all_time = 256.0 # {ms}

img = np.array([200,10])

def timing_coder(img):
    timelines_array = np.zeros_like([(np.arange(0,all_time,step_time)) for x in img])
    for i, value in enumerate(img):
        timelines_array[i][value] = 1.47
    return timelines_array

timelines = timing_coder(img)

V1 = 0
spike1 = 0
trace1 = 0
V1_list = []
spike1_list = []
trace1_list = []

V2 = 0
spike2 = 0
trace2 = 0
V2_list = []
spike2_list = []
trace2_list = []

V3 = 0
spike3 = 0
trace3 = 0
V3_list = []
spike3_list = []
trace3_list = []

w1 = 0.7
w1_list = []

w2 = 0.7
w2_list = []

def LIF(V,spike,trace,I):
    if spike:
        V -= 1.0
    V = V - (V/10.0) + I
    spike = 0
    if V >= 1.0:
        spike = 1
        trace += 1.0
    else:
        trace = trace - (trace/10.0)
    return V, spike, trace

def STDP(w,pre_spike,post_spike,pre_trace,post_trace):
    if pre_spike:
        w = w - 0.1 * w * post_trace
    if post_spike:
        w = w + 0.1 * (1 - w) * pre_trace
    return w

for epoch in range(10):
    for i in range(len(timelines[0])):
        V1,spike1,trace1 = LIF(V1,spike1,trace1,timelines[0][i])
        V1_list.append(V1)
        spike1_list.append(spike1)
        trace1_list.append(trace1)

        V2,spike2,trace2 = LIF(V2,spike2,trace2,timelines[1][i])
        V2_list.append(V2)
        spike2_list.append(spike2)
        trace2_list.append(trace2)

        V3,spike3,trace3 = LIF(V3,spike3,trace3,spike1*w1 + spike2*w2)
        V3_list.append(V3)
        spike3_list.append(spike3)
        trace3_list.append(trace3)

        w1 = STDP(w1,spike1,spike3,trace1,trace3)
        w1_list.append(w1)

        w2 = STDP(w2,spike2,spike3,trace2,trace3)
        w2_list.append(w2)

fig, axes = plt.subplots(nrows=3,ncols=3,figsize=(8,8))

axes[0,0].plot(V1_list)
axes[0,0].set_title("V1")

axes[0,1].plot(V2_list)
axes[0,1].set_title("V2")

axes[0,2].plot(V3_list)
axes[0,2].set_title("V3")

axes[1,0].plot(spike1_list)
axes[1,0].set_title("Spike1")

axes[1,1].plot(spike2_list)
axes[1,1].set_title("Spike2")

axes[1,2].plot(spike3_list)
axes[1,2].set_title("Spike3")

axes[2,0].plot(w1_list)
axes[2,0].set_title("W1")

axes[2,1].plot(w2_list)
axes[2,1].set_title("W2")

plt.show()