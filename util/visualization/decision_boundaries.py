import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def plot_decision_boundaries(coords, output_winners, output_confidence, grid_x, grid_y, point_x, point_y,
                             point_class, num_classes, step, writer=None):
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
    point_class += 1

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

    for i in np.unique(output_winners):
        locs = np.where(output_winners == i)
        grid_vals = griddata(coords[locs[0]], output_confidence[locs[0]], (grid_x, grid_y), method='cubic')
        grid_vals = np.flip(grid_vals, axis=0)
        axs.imshow(grid_vals, extent=(np.min(grid_x), np.max(grid_x), np.min(grid_y), np.max(grid_y)),
                   cmap=colors_contour[colors[i]])

    # Draw all the points
    for i in range(1, num_classes + 1):
        locs = np.where(point_class == i)
        axs.scatter(point_x[locs], point_y[locs], c=colors_points[colors[i - 1]], edgecolor='w', lw=0.75)

    # Draw image
    fig.canvas.draw()

    # Get image
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Plot to tensorboard
    writer.add_image('decision_boundary_overview', data, global_step=step)
    writer.add_image('decision_boundary/{}'.format(step), data, global_step=step)

    return None
