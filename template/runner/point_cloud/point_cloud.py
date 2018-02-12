"""
This file is the template for the boilerplate of train/test of a DNN on a points cloud dataset
In particular, point_cloud is designed to work with clouds of bi-dimensional points.

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""

# Utils

# Torch
from torch import nn

# DeepDIVA
import models
from init.initializer import *
from template.runner.point_cloud.evaluate import validate, test
from template.runner.point_cloud.train import train
from template.runner.standard import Standard
from template.setup import set_up_model, set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate
from util.visualization.decision_boundaries import plot_decision_boundaries


#######################################################################################################################
class PointCloud(Standard):

    @staticmethod
    def single_run(writer, log_dir, model_name, epochs, lr, decay_lr, **kwargs):
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

           :param decay_lr: boolean
                Decay the lr flag

           :param kwargs: dict
               Any additional arguments.

           :return: train_value, val_value, test_value
               Precision values for train and validation splits. Single precision value for the test split.
       """

        # Get the selected model
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(model_expected_input_size, **kwargs)

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=num_classes,
                                                                            model_name=model_name,
                                                                            lr=lr, **kwargs)

        # Core routine
        logging.info('Begin training')
        val_precs = np.zeros((epochs - start_epoch))
        train_precs = np.zeros((epochs - start_epoch))

        # Make data for points
        grid_resolution = 100
        coords_grid = np.array([[x, y] for x in np.linspace(0.0, 1.0, grid_resolution) for y in
                                np.linspace(0.0, 1.0, grid_resolution)])
        coords = torch.autograd.Variable(torch.from_numpy(coords_grid).type(torch.FloatTensor))

        if not kwargs['no_cuda']:
            coords = coords.cuda(async=True)

        PointCloud.evalute_and_plot_decision_boundary(model, coords, grid_resolution, val_loader, num_classes, writer,
                                                      epoch=-1, no_cuda=kwargs['no_cuda'])

        validate(val_loader, model, criterion, writer, -1, **kwargs)
        for epoch in range(start_epoch, epochs):
            # Train
            train_precs[epoch] = train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)
            # Validate
            val_precs[epoch] = validate(val_loader, model, criterion, writer, epoch, **kwargs)
            if decay_lr is not None:
                adjust_learning_rate(lr, optimizer, epoch, epochs)
            best_value = checkpoint(epoch, val_precs[epoch], best_value, model, optimizer, log_dir)

            # PLOT
            PointCloud.evalute_and_plot_decision_boundary(model, coords, grid_resolution, val_loader, num_classes,
                                                          writer, epoch=epoch, no_cuda=kwargs['no_cuda'])

        # Test
        test_prec = test(test_loader, model, criterion, writer, epoch, **kwargs)

        # PLOT
        PointCloud.evalute_and_plot_decision_boundary(model, coords, grid_resolution, val_loader, num_classes, writer,
                                                      epoch=epochs, no_cuda=kwargs['no_cuda'])

        logging.info('Training completed')

        return train_precs, val_precs, test_prec

    @staticmethod
    def evalute_and_plot_decision_boundary(model, coords, grid_resolution, val_loader, num_classes, writer, epoch,
                                           no_cuda):

        grid_x = np.linspace(0.0, 1.0, grid_resolution)
        grid_y = np.linspace(0.0, 1.0, grid_resolution)

        sm = nn.Softmax()

        if not no_cuda:
            outputs = model(coords)
            outputs = sm(outputs)
            outputs = outputs.data.cpu().numpy()
        else:
            outputs = sm(model(coords)).data.numpy()
        output_winners = np.array([np.argmax(item) for item in outputs])
        outputs = np.array([outputs[i, item] for i, item in enumerate(output_winners)])
        outputs = outputs + output_winners

        plot_decision_boundaries(grid_x, grid_y, outputs.reshape(len(grid_x), len(grid_x)),
                                 val_loader.dataset.data[:, 0], val_loader.dataset.data[:, 1],
                                 val_loader.dataset.data[:, 2], num_classes, step=epoch, writer=writer)
        return


