import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging


def plot_decision_boundaries(grid_x, grid_y, grid_z, point_x, point_y, point_class, num_classes, step, writer=None):
    """
    Plots the decision boundaries as a 2D image onto Tensorboard.
    :param grid_x: X axis locations of the decision grid
    :param grid_y: Y axis locations of the decision grid
    :param grid_z: classification values at each point on the decision grid
    :param point_x: X axis locations of the real points to be plotted
    :param point_y: Y axis locations of the real points to be plotted
    :param point_class: class of the real points at each location
    :param num_classes: number of unique classes
    :param step: global training step
    :param writer: Tensorboard summarywriter object
    :return: None
    """
    point_class = point_class.copy()
    X, Y = grid_x.copy(), grid_y.copy()
    zdata = grid_z.copy().T + 1
    point_class += 1

    levels = []
    for i in range(0, num_classes):
        levels.append(np.linspace(i + 1, i + 2, 1000))

    # Matplotlib stuff
    fig = plt.figure(1)
    axs = plt.gca()

    colors = ['blue', 'orange', 'red', 'green', 'purple']
    colors_points = {'blue': '#000099',
                     'orange': '#e68a00',
                     'red': '#b30000',
                     'green': '#009900',
                     'purple': '#7300e6'}
    colors_contour = {'blue': plt.get_cmap('Blues'),
                      'orange': plt.get_cmap('Oranges'),
                      'red': plt.get_cmap('Reds'),
                      'green': plt.get_cmap('Greens'),
                      'purple': plt.get_cmap('Purples')}

    # Identify which locations belong to which classes
    zdata_floor = np.floor(zdata)
    locs = np.where(zdata_floor == num_classes + 1)
    zdata_floor[locs[0], locs[1]] -= 0.001
    zdata_floor = np.floor(zdata_floor)

    # Draw all the decision boundaries
    for i in range(1, num_classes + 1):
        try:
            tmp = np.copy(zdata)
            locs = np.where(zdata_floor != i)
            tmp[locs[0], locs[1]] = 0
            locs = np.where(tmp != 0)
            vmin, vmax = np.min(tmp[locs[0], locs[1]]), np.max(tmp[locs[0], locs[1]])
            axs.contourf(Y, X, tmp, levels=levels[i - 1], cmap=colors_contour[colors[i - 1]], vmin=vmin, vmax=vmax)
        except ValueError:
            # TODO choose which of the 2 following lines :)
            continue
            logging.warning("No predictions for class {}".format(i - 1))

    # Draw all the points
    for i in range(1, num_classes + 1):
        locs = np.where(point_class == i)
        axs.scatter(point_x[locs], point_y[locs], c=colors_points[colors[i - 1]], edgecolor='w', lw=0.75)

    # Draw image
    fig.canvas.draw()

    # Get image
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Plot to visdom
    writer.add_image('decision_boundary_overview', data, global_step=step)
    writer.add_image('decision_boundary/{}'.format(step), data, global_step=step)

    return None
