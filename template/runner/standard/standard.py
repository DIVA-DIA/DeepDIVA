"""
This file is the template for the boilerplate of train/test of a DNN

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""

# DeepDIVA
import models
from init.initializer import *
# Delegated
from template.runner.standard import evaluate, train
from template.setup import set_up_model, set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate


#######################################################################################################################
class Standard:
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

        :param kwargs: dict
            Any additional arguments.

        :param decay_lr: boolean
                Decay the lr flag

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
        val_value = np.zeros((epochs - start_epoch))
        train_value = np.zeros((epochs - start_epoch))

        Standard._validate(val_loader, model, criterion, writer, -1, **kwargs)
        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = Standard._train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)
            # Validate
            val_value[epoch] = Standard._validate(val_loader, model, criterion, writer, epoch, **kwargs)
            if decay_lr is not None:
                adjust_learning_rate(lr, optimizer, epoch, epochs)
            best_value = checkpoint(epoch, val_value[epoch], best_value, model, optimizer, log_dir)

        # Test
        test_value = Standard._test(test_loader, model, criterion, writer, epochs - 1, **kwargs)
        logging.info('Training completed')

        return train_value, val_value, test_value

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
