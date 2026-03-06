from data_coder import timing_coder
from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Загружаем все изображения (пример: 100 цифр)
image_paths = [fr"D:\GitHub\Neural-network\src\datasets\unzip_datasets\mnist\images\image_{i}.jpg" for i in range(10)]  # ваши файлы
images = []
for path in image_paths:
    img = Image.open(path).convert('L')
    images.append(np.array(img))
images = np.array(images)  # форма (100, 28, 28)

allTime = 256 #ms
step_time = 1 #ms

timeLines = timing_coder(images)

l1 = np.array([Neuron(10) for _ in range(784)])
l2 = np.array([Neuron() for _ in range(10)])

for epoch in range(10):
    w_to_neuron0 = np.array([n.W[0] for n in l1])  # веса от всех 784 нейронов к выходному нейрону 0
    w_img = w_to_neuron0.reshape(28, 28)
    plt.imshow(w_img, cmap='hot')
    plt.savefig(f'w_{epoch}.png')
    for i in range(len(timeLines)):
        timeline_T = timeLines[i].T
        for n in l1:
            n.V = 0
            n.trace = 0
        for n in l2:
            n.V = 0
            n.trace = 0
        k = [0,0,0,0,0,0,0,0,0,0]
        for i in range(allTime):
            tick = np.array(timeline_T[i])
            for j, n in enumerate(l1):
                n.LIF(tick[j])
            for j, n in enumerate(l2):
                n.LIF(sum([x.out[j] for x in l1]))
                k[j] += n.spike
            for j, n in enumerate(l1):
                n.STDP(l2)


# fig, axes = plt.subplots(nrows=3,ncols=4,figsize=(12,12))

# axes[0,0].plot(l2[0].v_list)
# axes[0,0].set_title("V0")

# axes[0,1].plot(l2[1].v_list)
# axes[0,1].set_title("V1")

# axes[0,2].plot(l2[2].v_list)
# axes[0,2].set_title("V2")

# axes[0,3].plot(l2[3].v_list)
# axes[0,3].set_title("V3")

# axes[1,0].plot(l2[4].v_list)
# axes[1,0].set_title("V4")

# axes[1,1].plot(l2[5].v_list)
# axes[1,1].set_title("V5")

# axes[1,2].plot(l2[6].v_list)
# axes[1,2].set_title("V6")

# axes[1,3].plot(l2[7].v_list)
# axes[1,3].set_title("V7")

# axes[2,0].plot(l2[8].v_list)
# axes[2,0].set_title("V8")

# axes[2,1].plot(l2[9].v_list)
# axes[2,1].set_title("V9")

# plt.savefig('V_plot.png')

# fig, axes = plt.subplots(nrows=3,ncols=4,figsize=(12,12))

# axes[0,0].plot(l1[100].w_list)
# axes[0,0].set_title("V0")

# axes[0,1].plot(l1[409].w_list)
# axes[0,1].set_title("V1")

# axes[0,2].plot(l1[504].w_list)
# axes[0,2].set_title("V2")

# axes[0,3].plot(l1[56].w_list)
# axes[0,3].set_title("V3")

# axes[1,0].plot(l1[99].w_list)
# axes[1,0].set_title("V4")

# axes[1,1].plot(l1[490].w_list)
# axes[1,1].set_title("V5")

# axes[1,2].plot(l1[380].w_list)
# axes[1,2].set_title("V6")

# axes[1,3].plot(l1[250].w_list)
# axes[1,3].set_title("V7")

# axes[2,0].plot(l1[400].w_list)
# axes[2,0].set_title("V8")

# axes[2,1].plot(l1[280].w_list)
# axes[2,1].set_title("V9")

# plt.savefig('W_plot.png')