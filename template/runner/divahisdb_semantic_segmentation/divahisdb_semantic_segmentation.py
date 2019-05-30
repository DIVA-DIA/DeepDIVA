"""
This file is the template for the boilerplate of train/test of a DNN for image classification

There are a lot of parameter which can be specified to modify the behaviour and they should be used 
instead of hard-coding stuff.
"""

# Utils

# Delegated
from template.runner import ImageClassification
from template.runner.divahisdb_semantic_segmentation.setup import set_up_dataloaders
from template.setup import set_up_model
from . import evaluate, train


class DivahisdbSemanticSegmentation(ImageClassification):

    class_encoding = None
    img_names_sizes_dict = None

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
        # Setting up the dataloaders
        train_loader, val_loader, test_loader = set_up_dataloaders(**kwargs)
        cls.class_encoding = train_loader.dataset.class_encodings
        cls.img_names_sizes_dict = dict(test_loader.dataset.img_names_sizes)  # (gt_img_name, img_size (H, W))

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value = set_up_model(model_name=model_name,
                                                               num_classes=len(cls.class_encoding),
                                                               **kwargs)
        return model, len(cls.class_encoding), best_value, train_loader, val_loader, test_loader, optimizer, criterion

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _train(cls, **kwargs):
        return train.train(class_encodings=cls.class_encoding, **kwargs)

    @classmethod
    def _validate(cls, **kwargs):
        return evaluate.validate(class_encodings=cls.class_encoding, **kwargs)

    @classmethod
    def _test(cls, **kwargs):
        return evaluate.test(class_encodings=cls.class_encoding, img_names_sizes_dict=cls.img_names_sizes_dict, **kwargs)
