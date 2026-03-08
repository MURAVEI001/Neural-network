import matplotlib.pyplot as plt
import numpy as np

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