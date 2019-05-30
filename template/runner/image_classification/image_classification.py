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
import models
# Delegated
from template.runner.image_classification.train import train
from template.runner.image_classification.evaluate import evaluate
from template.setup import set_up_model, set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate


class ImageClassification:
    @classmethod
    def single_run(cls, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        test_value : float
            Accuracy value for test split
        """

        # Prepare the data, optimizer and criterion
        model, num_classes, best_value, train_loader, val_loader, test_loader, optimizer, criterion = cls.prepare(**kwargs)

        # Train routine
        train_value, val_value = cls.train_routine(model=model, best_value=best_value,
                                                   optimizer=optimizer, criterion=criterion,
                                                   train_loader=train_loader, val_loader=val_loader,
                                                   **kwargs)

        # Test routine
        test_value = cls.test_routine(criterion=criterion, num_classes=num_classes, test_loader=test_loader,
                                      **kwargs)

        return train_value, val_value, test_value


    ####################################################################################################################
    @classmethod
    def prepare(cls, model_name, **kwargs):
        """
        Loads and prepares the data, the optimizer and the criterion

        Parameters
        ----------
        model_name : str
            Name of the model. Used for loading the model.
        kwargs : dict
            Any additional arguments.

        Returns
        -------
        model : DataParallel
            The model to train
        num_classes : int
            How many different classes there are in our problem. Used for loading the model.
        best_value : float
            Best value of the model so far. Non-zero only in case of --resume being used
        train_loader : torch.utils.data.dataloader.DataLoader
            Training dataloader
        val_loader : torch.utils.data.dataloader.DataLoader
            Validation dataloader
        test_loader : torch.utils.data.dataloader.DataLoader
            Test set dataloader
        optimizer : torch.optim
            Optimizer to use during training, e.g. SGD
        criterion : torch.nn.modules.loss
            Loss function to use, e.g. cross-entropy
        """
        # Get the selected model input size
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        if type(model_expected_input_size) is not tuple or len(model_expected_input_size) != 2:
            logging.error('Model {model_name} expected input size is not a tuple. '
                          'Received: {model_expected_input_size}'
                          .format(model_name=model_name,
                                  model_expected_input_size=model_expected_input_size))
            sys.exit(-1)
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(model_expected_input_size, **kwargs)

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value = set_up_model(model_name=model_name, num_classes=num_classes,
                                                               **kwargs)
        return  model, num_classes, best_value, train_loader, val_loader, test_loader, optimizer, criterion

    @classmethod
    def train_routine(cls, best_value, decay_lr, validation_interval, start_epoch, epochs, checkpoint_all_epochs,
                      current_log_folder,
                      **kwargs):
        """
        Performs the training and validatation routines

        Parameters
        ----------
        best_value : float
            Best value of the model so far. Non-zero only in case of --resume being used
        decay_lr : boolean
            Decay the lr flag
        validation_interval : int
            Run evaluation on validation set every N epochs
        start_epoch : int
            Int to initialize the starting epoch. Non-zero only in case of --resume being used
        epochs : int
            Number of epochs to train
        checkpoint_all_epochs : bool
            Save checkpoint at each epoch
        current_log_folder : string
            Path to where logs/checkpoints are saved
        kwargs : dict
            Any additional arguments.

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        """
        logging.info('Begin training')
        val_value = np.zeros((epochs + 1 - start_epoch))
        train_value = np.zeros((epochs - start_epoch))

        # Validate before training
        val_value[-1] = cls._validate(epoch=-1, **kwargs)
        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = cls._train(epoch=epoch, **kwargs)

            # Validate
            if epoch % validation_interval == 0:
                val_value[epoch] = cls._validate(epoch=epoch, **kwargs)
            if decay_lr is not None:
                adjust_learning_rate(epoch=epoch, decay_lr_epochs=decay_lr, **kwargs)
            # Checkpoint
            best_value = checkpoint(epoch=epoch,
                                    new_value=val_value[epoch],
                                    best_value=best_value,
                                    log_dir=current_log_folder,
                                    checkpoint_all_epochs=checkpoint_all_epochs,
                                    **kwargs)
        logging.info('Training done')
        return train_value, val_value

    @classmethod
    def test_routine(cls, model_name, num_classes, criterion, epochs, current_log_folder, writer,
                     **kwargs):
        """
        Load the best model according to the validation score (early stopping) and runs the test routine.

        Parameters
        ----------
        model_name : str
            name of the model. Used for loading the model.
        num_classes : int
            How many different classes there are in our problem. Used for loading the model.
        criterion : torch.nn.modules.loss
            Loss function to use, e.g. cross-entropy
        epochs : int
            After how many epochs are we testing
        current_log_folder : string
            Path to where logs/checkpoints are saved
        writer : Tensorboard.SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        kwargs : dict
            Any additional arguments.

        Returns
        -------
        test_value : float
            Accuracy value for test split
        """
        # Load the best model before evaluating on the test set.
        logging.info('Loading the best model before evaluating on the test set.')
        kwargs["load_model"] = os.path.join(current_log_folder,
                                            'model_best.pth.tar')
        model, _, _, _ = set_up_model(num_classes=num_classes,
                                      model_name=model_name,
                                      **kwargs)
        # Test
        test_value = cls._test(model=model, criterion=criterion, writer=writer, epoch=epochs - 1, **kwargs)
        logging.info('Training completed')
        return test_value

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _train(cls, **kwargs):
        return train(**kwargs)

    @classmethod
    def _validate(cls, **kwargs):
        return evaluate(data_loader=kwargs['val_loader'], logging_label='val', **kwargs)

    @classmethod
    def _test(cls, **kwargs):
        return evaluate(data_loader=kwargs['test_loader'], logging_label='test', **kwargs)
