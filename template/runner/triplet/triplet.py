"""
This file is the template for the boilerplate of train/test of a triplet network.
This code has initially been adapted to our purposes from
http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
"""

# Utils
from __future__ import print_function
import logging
import sys
import numpy as np
import torch
import torch.nn as nn

# DeepDIVA
import models
from torch.nn import init
from template.runner.triplet.setup import setup_dataloaders
from template.setup import set_up_model
from util.misc import adjust_learning_rate, checkpoint

# Delegated
from template.runner.triplet import train, evaluate


#######################################################################################################################


class Triplet:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr,
                   margin, anchor_swap, validation_interval, regenerate_every,
                   checkpoint_all_epochs, only_evaluate, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Parameters
        ----------
        writer : Tensorboard SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        current_log_folder : string
            Path to where logs/checkpoints are saved
        model_name : string
            Name of the model
        epochs : int
            Number of epochs to train
        lr : float
            Value for learning rate
        margin : float
            The margin value for the triplet loss function
        anchor_swap : boolean
            Turns on anchor swap
        decay_lr : boolean
            Decay the lr flag
        validation_interval : int
            Run evaluation on validation set every N epochs
        regenerate_every : int
            Re-generate triplets every N epochs
        checkpoint_all_epochs : bool
            If enabled, save checkpoint after every epoch.
        only_evaluate : boolean
            Flag : if True, only the test set is loaded.

        Returns
        -------
        train_value, val_value, test_value
            Mean Average Precision values for train and validation splits.
        """
        # Sanity check on parameters
        if kwargs["output_channels"] is None:
            logging.error("Using triplet class but --output-channels is not specified.")
            sys.exit(-1)

        # Get the selected model input size
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        Triplet._validate_model_input_size(model_expected_input_size, model_name)
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        if only_evaluate:
            _, _, test_loader = setup_dataloaders(
                model_expected_input_size=model_expected_input_size,
                only_evaluate=only_evaluate,
                **kwargs)
        else:
            train_loader, val_loader, test_loader = setup_dataloaders(
                model_expected_input_size=model_expected_input_size,
                **kwargs)

        # Setting up model, optimizer, criterion
        model, _, optimizer, best_value, start_epoch = set_up_model(model_name=model_name,
                                                                    lr=lr,
                                                                    **kwargs)

        # Set the special criterion for triplets
        criterion = nn.TripletMarginLoss(margin=margin, swap=anchor_swap)

        train_value = np.zeros((epochs - start_epoch))
        val_value = np.zeros((epochs - start_epoch))

        if not only_evaluate:
            # Core routine

            logging.info('Begin training')
            Triplet._validate(val_loader, model, None, writer, -1, **kwargs)
            for epoch in range(start_epoch, epochs):
                # Train
                train_value[epoch] = Triplet._train(train_loader=train_loader,
                                                    model=model,
                                                    criterion=criterion,
                                                    optimizer=optimizer,
                                                    writer=writer,
                                                    epoch=epoch,
                                                    **kwargs)
                # Validate
                if epoch % validation_interval == 0:
                    val_value[epoch] = Triplet._validate(val_loader=val_loader,
                                                         model=model,
                                                         writer=writer,
                                                         epoch=epoch,
                                                         **kwargs)
                if decay_lr is not None:
                    adjust_learning_rate(lr, optimizer, epoch, epochs)
                best_value = checkpoint(epoch=epoch,
                                        new_value=val_value[epoch],
                                        best_value=best_value,
                                        model=model,
                                        optimizer=optimizer,
                                        log_dir=current_log_folder,
                                        invert_best=True,
                                        checkpoint_all_epochs=checkpoint_all_epochs)

                # Generate new triplets every N epochs
                if epoch % regenerate_every == 0:
                    train_loader.dataset.generate_triplets()
            logging.info('Training completed')

        # Test
        test_value = Triplet._test(test_loader=test_loader,
                                   model=model,
                                   writer=writer,
                                   epoch=(epochs - 1),
                                   **kwargs)

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
    # These methods delegate their function to other classes in this package.
    # It is useful because sub-classes can selectively change the logic of certain parts only.

    @classmethod
    def _train(cls, train_loader, model, criterion, optimizer, writer, epoch, **kwargs):
        return train.train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

    @classmethod
    def _validate(cls, val_loader, model, writer, epoch, **kwargs):
        return evaluate.validate(val_loader, model, writer, epoch, **kwargs)

    @classmethod
    def _test(cls, test_loader, model, writer, epoch, **kwargs):
        return evaluate.test(test_loader, model, writer, epoch, **kwargs)
