import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_all_weights(W, epoch, save_path='.', vmin=0, vmax=1, colorbar=True):

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    im = None
    W = W.detach().cpu().numpy()
    for i, ax in enumerate(axes.flat):
        # Преобразование столбца весов в изображение 28×28
        w_to_neuron = W[:, i].reshape(28, 28)
        im = ax.imshow(w_to_neuron, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(i)
        ax.axis('off')
    
    if colorbar and im is not None:
        # Общий цветовой бар для всех subplots
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Значение веса')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(f'{save_path}/weights_epoch_{epoch:03d}.png')