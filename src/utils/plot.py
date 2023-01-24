import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')

def plot_losses(train_losses, val_losses, dest: str = None):
    plt.figure(figsize=(15, 5))
    n_epochs = len(val_losses)
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, val_losses, label='val loss')
    plt.legend()
    plt.title('Training Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy + Dice')
    if dest:
        plt.savefig(dest)
    plt.tight_layout()

    
    # plt.savefig('segmentation_plot.png')

def plot_generic(title: str, xlabel: str, label1: str, data1: list, label2: str, data2: list, dest: str = None):
    n_epochs = len(data2)
    plt.figure(figsize=(9, 3))
    plt.plot(np.arange(1, n_epochs + 1), data1, label=label1)
    plt.plot(np.arange(1, n_epochs + 1), data2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('')
    plt.legend()
    if dest:
        plt.savefig(dest)
    plt.show()
