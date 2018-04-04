import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_mean_std(x=None, arr=None, suptitle='', title='', xlabel='X', ylabel='Y', xlim=None, ylim=None):
    fig = plt.figure(1)
    arr_mean = np.mean(arr, 0)
    arr_std = np.std(arr, 0)
    arr_min = np.min(arr, 0)
    arr_max = np.max(arr, 0)
    with sns.axes_style('darkgrid'):
        fig.suptitle(suptitle)
        plt.title(title)
        axes = plt.gca()
        if ylim is not None:
            axes.set_ylim(ylim)
        if xlim is not None:
            axes.set_xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if x is None:
            plt.plot(arr_mean, '-', color='#0000b3', label='Score')
            plt.plot(arr_min, color='#4d4dff', linestyle='dashed', label='Min')
            plt.plot(arr_max, color='#4d4dff', linestyle='dashed', label='Max')
            axes.fill_between(np.arange(len(arr_mean)), arr_mean - arr_std, arr_mean + arr_std, color='#9999ff',
                              alpha=0.2)
        else:
            plt.plot(x, arr_mean, '-', color='#0000b3', label='Score')
            plt.plot(x, arr_min, color='#4d4dff', linestyle='dashed', label='Min')
            plt.plot(x, arr_max, color='#4d4dff', linestyle='dashed', label='Max')
            axes.fill_between(np.arange(len(arr_mean)) - 1, arr_mean - arr_std, arr_mean + arr_std, color='#9999ff',
                              alpha=0.2)
        plt.legend(loc='best')

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        plt.close()
    return data
