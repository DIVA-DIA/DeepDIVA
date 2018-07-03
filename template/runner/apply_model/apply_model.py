"""
This file is the template for the boilerplate of train/test of a DNN

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""

import logging
import os
import pickle

# DeepDIVA
import models
# Delegated
from template.runner.apply_model import evaluate
from template.runner.apply_model.setup import set_up_dataloader
from template.setup import set_up_model


# Utils


#######################################################################################################################
class ApplyModel:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr,
                   output_channels, classify, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Parameters
        ----------
        writer: Tensorboard SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.

        current_log_folder: string
            Path to where logs/checkpoints are saved

        model_name: string
            Name of the model

        epochs: int
            Number of epochs to train

        lr: float
            Value for learning rate

        kwargs: dict
            Any additional arguments.

        decay_lr: boolean
            Decay the lr flag

        output_channels: int
            Specify shape of final layer of network.

        classify : boolean
            Specifies whether to generate a classification report for the data or not.

        Returns
        -------
        None: None
            None
        """

        # Get the selected model input size
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        data_loader, num_classes = set_up_dataloader(model_expected_input_size=model_expected_input_size,
                                                     classify=classify, **kwargs)

        # Setting up model, optimizer, criterion
        output_channels = num_classes if classify else output_channels

        model, _, _, _, _ = set_up_model(output_channels=output_channels,
                                         model_name=model_name,
                                         lr=lr,
                                         train_loader=None,
                                         **kwargs)

        logging.info('Apply model to dataset')
        results = ApplyModel._feature_extract(writer=writer,
                                              data_loader=data_loader,
                                              model=model,
                                              epoch=-1,
                                              classify=classify,
                                              **kwargs)
        with open(os.path.join(current_log_folder, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        return None, None, None

    ####################################################################################################################
    # These methods delegate their function to other classes in this package.
    # It is useful because sub-classes can selectively change the logic of certain parts only.

    @classmethod
    def _feature_extract(cls, writer, data_loader, model, epoch, **kwargs):
        return evaluate.feature_extract(writer=writer, data_loader=data_loader, model=model, epoch=epoch, **kwargs)
