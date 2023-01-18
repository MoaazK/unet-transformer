import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def plot_losses(train_losses, val_losses):
    plt.figure()
    n_epochs = len(val_losses)
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, val_losses, label='val loss')
    plt.legend()
    plt.title('Training Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.tight_layout()
    # plt.savefig('segmentation_plot.png')