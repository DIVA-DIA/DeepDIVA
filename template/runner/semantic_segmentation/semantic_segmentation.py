"""
This file is the template for the boilerplate of train/test of a DNN for image classification

There are a lot of parameter which can be specified to modify the behaviour and they should be used 
instead of hard-coding stuff.
"""

import logging
import sys
import os

# Utils
import numpy as np

# DeepDIVA
from torch import nn

import models
# Delegated
from template.runner.semantic_segmentation import evaluate, train
from template.setup import set_up_model
from .setup import set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate


class SemanticSegmentation:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr,
                   validation_interval, checkpoint_all_epochs,
                   input_patch_size, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Parameters
        ----------
        writer : Tensorboard.SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        current_log_folder : string
            Path to where logs/checkpoints are saved
        model_name : string
            Name of the model
        epochs : int
            Number of epochs to train
        lr : float
            Value for learning rate
        kwargs : dict
            Any additional arguments.
        decay_lr : boolean
            Decay the lr flag
        validation_interval : int
            Run evaluation on validation set every N epochs
        checkpoint_all_epochs : bool
            If enabled, save checkpoint after every epoch.
        input_patch_size : int
            Size of the input patch, e.g. with 32 the input will be re-sized to 32x32

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        test_value : float
            Accuracy value for test split
        """

        # Setting up the dataloaders
        train_loader, val_loader, test_loader = set_up_dataloaders(input_patch_size, **kwargs)

        # Setting up model, optimizer, criterion
        model, _, optimizer, best_value, start_epoch = set_up_model(num_classes=3, # In this case is the num dimension of the output
                                                                    model_name=model_name,
                                                                    lr=lr,
                                                                    train_loader=train_loader,
                                                                    **kwargs)

        criterion = nn.MSELoss()

        # Core routine
        logging.info('Begin training')
        val_value = np.zeros((epochs + 1 - start_epoch))
        train_value = np.zeros((epochs - start_epoch))

        val_value[-1] = SemanticSegmentation._validate(val_loader, model, criterion, writer, -1, **kwargs)
        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = SemanticSegmentation._train(train_loader, model, criterion, optimizer, writer, epoch,
                                                             **kwargs)

            # Validate
            if epoch % validation_interval == 0:
                val_value[epoch] = SemanticSegmentation._validate(val_loader, model, criterion, writer, epoch, **kwargs)
            if decay_lr is not None:
                adjust_learning_rate(lr=lr, optimizer=optimizer, epoch=epoch, decay_lr_epochs=decay_lr)
            best_value = checkpoint(epoch=epoch, new_value=val_value[epoch],
                                    best_value=best_value, model=model,
                                    optimizer=optimizer,
                                    log_dir=current_log_folder,
                                    checkpoint_all_epochs=checkpoint_all_epochs)


        # Load the best model before evaluating on the test set.
        logging.info('Loading the best model before evaluating on the test set.')
        kwargs["load_model"] = os.path.join(current_log_folder, 'model_best.pth.tar')
        model, _, _, _, _ = set_up_model(num_classes=3,
                                         model_name=model_name,
                                         lr=lr,
                                         train_loader=train_loader,
                                         **kwargs)

        # Test
        test_value = SemanticSegmentation._test(test_loader, model, criterion, writer, epochs - 1, **kwargs)
        logging.info('Training completed')

        return train_value, val_value, test_value

    ####################################################################################################################
    @staticmethod
    def _validate_model_input_size(model_expected_input_size, model_name):
        """
        This method verifies that the model expected input size is a tuple of 2 elements.
        This is necessary to avoid confusion with models which run on other types of data.

        Parameters
        ----------
        model_expected_input_size
            The item retrieved from the model which corresponds to the expected input size
        model_name : String
            Name of the model (logging purpose only)

        Returns
        -------
            None
        """
        if type(model_expected_input_size) is not tuple or len(model_expected_input_size) != 2:
            logging.error('Model {model_name} expected input size is not a tuple. '
                          'Received: {model_expected_input_size}'
                          .format(model_name=model_name,
                                  model_expected_input_size=model_expected_input_size))
            sys.exit(-1)

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _train(cls, train_loader, model, criterion, optimizer, writer, epoch, **kwargs):
        return train.train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

    @classmethod
    def _validate(cls, val_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.validate(val_loader, model, criterion, writer, epoch, **kwargs)

    @classmethod
    def _test(cls, test_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.test(test_loader, model, criterion, writer, epoch, **kwargs)
