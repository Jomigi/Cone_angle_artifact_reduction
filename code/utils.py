import matplotlib.pyplot as plt
import numpy as np# linear algebra


def plot_loss(epochs, results_folder, loss_list, loss_f, dataset):
    x_grid = list(range(epochs))
    plt.title(dataset + ' Losses Across Epochs')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')
    plt.xticks(x_grid)
    plt.plot(loss_list)
    fig_path = results_folder + dataset + 'loss'
    plt.savefig(fig_path)
    #plt.show()
    plt.close('all')
