"""
This file is the template for the boilerplate of train/test of a DNN on a points cloud dataset
In particular, point_cloud is designed to work with clouds of bi-dimensional points.

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""


# Utils
import os

import datasets
# DeepDIVA
import models
from init.initializer import *
from template.runner import Standard
from template.runner.point_cloud.evaluate import validate, test
from template.runner.point_cloud.train import train
from template.setup import set_up_model
from util.misc import checkpoint, adjust_learning_rate
from util.visualization.point_cloud import plot_to_visdom


#######################################################################################################################
class PointCloud(Standard):
    @staticmethod
    def single_run(writer, log_dir, model_name, epochs, lr, **kwargs):
        """
           This is the main routine where train(), validate() and test() are called.

           Parameters
           ----------
           :param writer: Tensorboard SummaryWriter
               Responsible for writing logs in Tensorboard compatible format.

           :param log_dir: string
               Path to where logs/checkpoints are saved

           :param model_name: string
               Name of the model

           :param epochs: int
               Number of epochs to train

           :param lr: float
               Value for learning rate

           :param kwargs: dict
               Any additional arguments.

           :return: train_value, val_value, test_value
               Precision values for train and validation splits. Single precision value for the test split.
       """

        # Get the selected model
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes = PointCloud.set_up_dataloaders(**kwargs)

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=num_classes,
                                                                            model_name=model_name,
                                                                            lr=lr, **kwargs)

        # Core routine
        logging.info('Begin training')
        val_value = np.zeros((epochs - start_epoch))
        train_value = np.zeros((epochs - start_epoch))

        # Make data for points
        POINTS_RESOLUTION = 100
        min_x, min_y = train_loader.dataset.min_coords
        max_x, max_y = train_loader.dataset.max_coords
        coords_np = np.array([[x, y] for x in np.linspace(min_x, max_x, POINTS_RESOLUTION) for y in
                              np.linspace(min_y, max_y, POINTS_RESOLUTION)])
        grid_x, grid_y = np.linspace(min_x, max_x, POINTS_RESOLUTION), np.linspace(min_y, max_y, POINTS_RESOLUTION)
        coords = torch.autograd.Variable(torch.from_numpy(coords_np).type(torch.FloatTensor))

        if not kwargs['no_cuda']:
            coords = coords.cuda(async=True)

        if not kwargs['no_cuda']:
            outputs = model(coords).data.cpu().numpy()
        else:
            outputs = model(coords).data.numpy()
        output_winners = np.array([np.argmax(item) for item in outputs])
        outputs = np.array([outputs[i, item] for i, item in enumerate(output_winners)])
        outputs = outputs + output_winners

        win_name = plot_to_visdom(grid_x, grid_y, outputs.reshape(len(grid_x), len(grid_x)),
                                  val_loader.dataset.data[:, 0],
                                  val_loader.dataset.data[:, 1], val_loader.dataset.data[:, 2], num_classes,
                                  win_name=None,
                                  writer=writer)

        validate(val_loader, model, criterion, writer, -1, **kwargs)
        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)
            # Validate
            val_value[epoch] = validate(val_loader, model, criterion, writer, epoch, **kwargs)
            if kwargs["decay_lr"] is not None:
                adjust_learning_rate(lr, optimizer, epoch, kwargs['decay_lr'])
            best_value = checkpoint(epoch, val_value[epoch], best_value, model, optimizer, log_dir)

            # PLOT

            if not kwargs['no_cuda']:
                outputs = model(coords).data.cpu().numpy()
            else:
                outputs = model(coords).data.numpy()
            output_winners = np.array([np.argmax(item) for item in outputs])
            outputs = np.array([outputs[i, item] for i, item in enumerate(output_winners)])
            outputs = outputs + output_winners

            win_name = plot_to_visdom(grid_x, grid_y, outputs.reshape(len(grid_x), len(grid_x)),
                                      val_loader.dataset.data[:, 0], val_loader.dataset.data[:, 1],
                                      val_loader.dataset.data[:, 2], num_classes, win_name=None, writer=writer)

        # Test
        test_value = test(test_loader, model, criterion, writer, epoch, **kwargs)

        # PLOT
        if not kwargs['no_cuda']:
            outputs = model(coords).data.cpu().numpy()
        else:
            outputs = model(coords).data.numpy()
        output_winners = np.array([np.argmax(item) for item in outputs])
        outputs = np.array([outputs[i, item] for i, item in enumerate(output_winners)])
        outputs = outputs + output_winners

        win_name = plot_to_visdom(grid_x, grid_y, outputs.reshape(len(grid_x), len(grid_x)),
                                  val_loader.dataset.data[:, 0],
                                  val_loader.dataset.data[:, 1], val_loader.dataset.data[:, 2], num_classes,
                                  win_name=None,
                                  writer=writer)

        logging.info('Training completed')

        return train_value, val_value, test_value

    #######################################################################################################################
    @staticmethod
    def set_up_dataloaders(dataset_folder, batch_size, workers, **kwargs):
        """
        Set up the dataloaders for the specified datasets.

        :param dataset_folder : string
            Path to the dataset

        :param batch_size: int
            Number of datapoints to process at once

        :param workers: int
            Number of workers to use for the dataloaders

        :param kwargs: dict
            Any additional arguments.

        :return: dataloader, dataloader, dataloader, int
            Three dataloaders for train, val and test. Number of classes for the model.
        """

        logging.info('Loading datasets')
        train_ds = datasets.point_cloud(path=os.path.join(dataset_folder, 'train', 'data.csv'))
        val_ds = datasets.point_cloud(path=os.path.join(dataset_folder, 'val', 'data.csv'))
        test_ds = datasets.point_cloud(path=os.path.join(dataset_folder, 'test', 'data.csv'))

        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   shuffle=True,
                                                   batch_size=batch_size,
                                                   num_workers=workers,
                                                   pin_memory=False)

        val_loader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=batch_size,
                                                 num_workers=workers,
                                                 pin_memory=False)

        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=batch_size,
                                                  num_workers=workers,
                                                  pin_memory=False)

        return train_loader, val_loader, test_loader, train_ds.num_classes
