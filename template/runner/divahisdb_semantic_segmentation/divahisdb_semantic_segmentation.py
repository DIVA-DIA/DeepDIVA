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
        See parent class for documentation
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
    @classmethod
    def _train(cls, **kwargs):
        return train.train(class_encodings=cls.class_encoding, **kwargs)

    @classmethod
    def _validate(cls, **kwargs):
        return evaluate.validate(class_encodings=cls.class_encoding, **kwargs)

    @classmethod
    def _test(cls, **kwargs):
        return evaluate.test(class_encodings=cls.class_encoding, img_names_sizes_dict=cls.img_names_sizes_dict, **kwargs)
