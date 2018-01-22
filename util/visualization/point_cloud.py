import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import visdom
import numpy as np
import logging


def plot_to_visdom(grid_x, grid_y, grid_z, point_x, point_y, point_class, num_classes, win_name=None, writer=None):
    """
    Plots decision landscape to Visdom
    :param grid_x:
    :param grid_y:
    :param grid_z:
    :param point_x:
    :param point_y:
    :param point_class:
    :param num_classes:
    :param win_name:
    :return:
    """
    # Forgive the hack, but it's a call by reference and point_class really shouldn't be modified.
    point_class = point_class.copy()
    # try:
    #     vis = visdom.Visdom()
    # except:
    #     logging.error('Visdom Server unavailable')
    X, Y = grid_x.copy(), grid_y.copy()
    zdata = grid_z.copy().T + 1
    point_class += 1

    # levels = np.linspace(1, num_classes + 1, 1000)
    levels = []
    for i in range(0, num_classes):
        levels.append(np.linspace(i+1, i + 2, 1000))
    # Matplotlib stuff
    fig = plt.figure(1)
    axs = plt.gca()

    # Plot [BLUE, ORANGE, RED, GREEN, PURPLE]
    colors_points = ['#000099', '#e68a00', '#b30000', '#009900', '#7300e6']
    # colors_contour = ['#ff4d4d', '#33ff33', '#4d4dff', '#bf80ff', '#ffcc80']
    colors_contour = [plt.get_cmap('Blues'), plt.get_cmap('Oranges'), plt.get_cmap('Reds'), plt.get_cmap('Greens'),
                      plt.get_cmap('Purples')]

    zdata_floor = np.floor(zdata)
    locs = np.where(zdata_floor == num_classes + 1)
    zdata_floor[locs[0], locs[1]] -= 0.001
    zdata_floor = np.floor(zdata_floor)

    for i in range(1, num_classes + 1):
        try:
            tmp = np.copy(zdata)
            locs = np.where(zdata_floor != i)
            tmp[locs[0], locs[1]] = 0
            locs = np.where(tmp != 0)
            vmin, vmax = np.min(tmp[locs[0], locs[1]]), np.max(tmp[locs[0], locs[1]])
            axs.contourf(Y, X, tmp, levels=levels[i - 1], cmap=colors_contour[i - 1], vmin=vmin, vmax=vmax)
        except ValueError:
            continue
            logging.warning("No predictions for class {}".format(i - 1))

    for i in range(1, num_classes + 1):
        locs = np.where(point_class == i)
        axs.scatter(point_x[locs], point_y[locs], c=colors_points[i - 1], edgecolor='w', lw=0.75)

    # Draw image
    fig.canvas.draw()
    # Get image
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Plot to visdom
    # win_name = vis.image(np.transpose(data, [2, 0, 1]), win=win_name)
    writer.add_image('decision_boundary', data)

    return None


if __name__ == "__main__":
    # Make data
    # x = np.linspace(0, 1, 100)
    min_x = 0
    min_y = 0
    max_x = 1
    max_y = 1
    POINTS_RESOLUTION = 100
    coords = np.array([[x, y] for x, y in zip(np.linspace(min_x, max_x, POINTS_RESOLUTION),
                                              np.linspace(min_y, max_y, POINTS_RESOLUTION))])

    # X,Y = np.meshgrid(coords, x.T)
    zdata = np.array([(x, y, np.round(x)) for x in np.linspace(0, 4, 100) for y in np.linspace(0, 1, 100)])
    zdata = zdata[:, 2].reshape(100, 100) + np.reshape(np.array([y
                                                                 for x in np.linspace(0, 1, 100)
                                                                 for y in np.linspace(0, 1, 100)]), (100, 100))

    plot_to_visdom(coords[:, 0], coords[:, 1], zdata, np.array([0.8, 0.6, 0.2, 0.6]), np.array([0.4, 0.6, 0.8, 0.2]),
                   np.array([0, 0, 1, 1]), 5)
