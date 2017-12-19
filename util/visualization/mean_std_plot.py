import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_mean_variance(arr, suptitle='', title='', xlabel='X', ylabel='Y', ylim=None):
    fig = plt.figure(1)
    arr_mean =  np.mean(arr, 0)
    arr_std = np.std(arr, 0)
    with sns.axes_style('darkgrid'):
        fig.suptitle(suptitle)
        plt.title(title)
        axes = plt.gca()
        if ylim is not None:
            axes.set_ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(arr_mean, 'b-', label='MeanIU (LDA)')
        axes.fill_between(np.arange(len(arr_mean)), arr_mean - arr_std, arr_mean + arr_std, color='blue', alpha=0.2)
        plt.legend(loc='best')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        plt.close()
    return data