from data_coder import timing_coder
from neuron import Node
import numpy as np
import matplotlib.pyplot as plt

image_path = [fr"D:\GitHub\Neural-network\src\datasets\unzip_datasets\mnist\images\image_{i}.jpg" for i in range(1)]

allTime = 256 #ms
step_time = 1 #ms

timeLines = timing_coder(image_path,allTime,power=1)

node1 = Node(784,10)

def plot_all_weights(W, epoch, save_path='.'):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        w_to_neuron = np.array(1-W[:,i]).reshape(28, 28)
        im = ax.imshow(w_to_neuron, cmap='Greys', interpolation='nearest')
        ax.set_title(i)
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'{save_path}/weights_epoch_{epoch:03d}.png')

for epoch in range(200):
    print(epoch)
    k = [0,0,0,0,0,0,0,0,0,0]
    for timeline in timeLines:
        for _ in range(20):
            node1.step(timeline)
            k += node1.layer_post.Spike
    print(k)
    plot_all_weights(node1.W, epoch)
#plot_all_weights(node1.W, 1)