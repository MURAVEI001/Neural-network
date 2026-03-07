from data_coder import timing_coder
from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt

image_path = [fr"D:\GitHub\Neural-network\src\datasets\unzip_datasets\mnist\images\image_{i}.jpg" for i in range(20)]

allTime = 256 #ms
step_time = 1 #ms

timeLines = timing_coder(image_path,allTime,power=5)

l1 = np.array([Neuron(10) for _ in range(784)])
l2 = np.array([Neuron() for _ in range(10)])


def plot_all_weights(l1, epoch, save_path='.'):
    """Сохраняет одно изображение с картами весов для всех 10 выходных нейронов."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        # Собираем веса от всех 784 нейронов первого слоя к i-му выходному нейрону
        w_to_neuron = np.array([n.W[i] for n in l1]).reshape(28, 28)
        im = ax.imshow(w_to_neuron, cmap='Greys', interpolation='nearest')
        ax.set_title(i)
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'{save_path}/weights_epoch_{epoch:03d}.png')

for epoch in range(200):
    for i in range(timeLines.shape[0]):
        print(i)
        timeline = timeLines[i]
        for n in l1:
            n.V = 0
            n.trace = 0
        for n in l2:
            n.V = 0
            n.trace = 0
            n.threshold = 1.0

        winner_found = False
        winner_index = -1
        for k in range(allTime):
            tick = np.array(timeline[k])
            for j, n in enumerate(l1):
                n.LIF(tick[j])
            for j, n in enumerate(l2):
                n.LIF(sum([x.out[j] for x in l1]))
            if not winner_found:
                spiked = [i for i, n in enumerate(l2) if n.spike]
                if len(spiked) > 0:
                                # Выбираем первого спайкнувшего (можно случайного, если их несколько на одном шаге)
                    winner_index = np.random.choice(spiked)
                    winner_found = True
                                # На текущем шаге подавляем остальных
                    for i in spiked:
                        if i != winner_index:
                            l2[i].spike = 0
                            l2[i].V = 0
            else:
                            # Победитель уже есть, подавляем всех остальных
                for i, n in enumerate(l2):
                    if i != winner_index:
                        n.spike = 0
                        n.V = 0
            for n in l1:
                n.STDP(l2)

    plot_all_weights(l1, epoch)

    fig, axes = plt.subplots(nrows=3,ncols=4,figsize=(12,12))

    axes[0,0].plot(l2[0].v_list)
    axes[0,0].set_title("V0")

    axes[0,1].plot(l2[1].v_list)
    axes[0,1].set_title("V1")

    axes[0,2].plot(l2[2].v_list)
    axes[0,2].set_title("V2")

    axes[0,3].plot(l2[3].v_list)
    axes[0,3].set_title("V3")

    axes[1,0].plot(l2[4].v_list)
    axes[1,0].set_title("V4")

    axes[1,1].plot(l2[5].v_list)
    axes[1,1].set_title("V5")

    axes[1,2].plot(l2[6].v_list)
    axes[1,2].set_title("V6")

    axes[1,3].plot(l2[7].v_list)
    axes[1,3].set_title("V7")

    axes[2,0].plot(l2[8].v_list)
    axes[2,0].set_title("V8")

    axes[2,1].plot(l2[9].v_list)
    axes[2,1].set_title("V9")

    plt.savefig('V_plot.png')
