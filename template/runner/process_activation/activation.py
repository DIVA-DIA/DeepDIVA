import logging
import os
import json
import uuid
from torchvision.utils import save_image

import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm


class Activation:
    def __init__(self, log_folder, model_name, dataset, process_size, save_cover, no_cuda):
        """
        Actication class

        Parameters
        ----------
        log_folder : string
            the DeepDIVA common log_folder path.
        model_name : string
            Name of the model.
        dataset : torch.nn.dataset
            Dataset prepared by DeepDIVA.
        process_size : int
            Number of item (picture of the dataset) processed.
        save_cover : boolean
            Save or not cover for classes and item processed.
        no_cuda : bool
            Specify whether to use the GPU or not.
        """
        self.log_folder = os.path.realpath(os.path.join(log_folder, 'activations'))
        self.cover_folder = os.path.join(self.log_folder, 'cover')
        self.data_folder = os.path.join(self.log_folder, 'data')

        self.model_name = model_name
        self.dataset = dataset
        self.sample_image = None

        self.process_size = process_size
        self.save_cover = save_cover
        self.no_cuda = no_cuda

        self.store = OrderedDict()

    def init(self, model):
        """
        This method initialize internal global according to model passed.
        Storage and create custom folders on the disk.

        Parameters
        ----------
        model : Torch.nn.model
            PyTorch model initialized.

        Returns
        -------
            None
        """
        logging.info('Creating activation directories')

        os.mkdir(self.log_folder)
        os.mkdir(self.data_folder)
        if self.save_cover:
            os.mkdir(self.cover_folder)

        logging.info('Init manifest structure')
        # Get sample dataset image
        self.sample_image = next(enumerate(self.dataset))[1][0]
        if not self.no_cuda:
            self.sample_image = Variable(self.sample_image.cuda())

        # Extract model's shape
        shape = Activation._capture_activations(model, self.sample_image, self.no_cuda, False)

        # Init value in global store
        self.store['datetime'] = datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        self.store['version'] = 2

        self.store['model'] = OrderedDict()
        self.store['model']['name'] = self.model_name
        self.store['model']['layers'] = shape

        self.store['items'] = OrderedDict()
        self.store['epochs'] = OrderedDict()

    def resolve_items(self):
        """
        This method prepare all the items to process, prepare and save
        cover for items (if needed), create the model's shape internally.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
        logging.info('Resolving process items and class')

        single = OrderedDict()
        classe = OrderedDict()

        general = OrderedDict()
        general[0] = OrderedDict()
        general[0]['key'] = uuid.uuid4().hex
        general[0]['class'] = -1
        general[0]['size'] = 0
        if self.save_cover:
            pass
            # default icon ? how to ?

        for i, (image, label) in enumerate(self.dataset):
            # break policy
            if i >= self.process_size:
                break

            item_index = str(i)
            image = image[0]
            label = int(Variable(label)[0])

            single[item_index] = OrderedDict()
            single[item_index]['key'] = uuid.uuid4().hex
            single[item_index]['class'] = label
            single[item_index]['size'] = 1
            if self.save_cover:
                cover_name = uuid.uuid4().hex + '.jpg'
                single[item_index]['cover'] = cover_name
                save_image(image, os.path.join(self.cover_folder, cover_name))

            if not label in classe:
                classe[label] = OrderedDict()
                classe[label]['key'] = uuid.uuid4().hex
                classe[label]['class'] = label
                classe[label]['size'] = 1
                if self.save_cover:
                    cover_name = uuid.uuid4().hex + '.jpg'
                    classe[label]['cover'] = cover_name
                    save_image(image, os.path.join(self.cover_folder, cover_name))
            else:
                classe[label]['size'] += 1

            general[0]['size'] += 1

        self.store['items']['single'] = single
        self.store['items']['class'] = classe
        self.store['items']['general'] = general

        self._save()

    def add_epoch(self, epoch_number, epoch_accuracy, model):
        """
        This method collect, compute and save all activation data (and mean activation
        data) from a given epoch

        Parameters
        ----------
        epoch_number : int
            Epoch number of the processing.
        epoch_accuracy : int
            Epoch accuracy retrived by the last training.
        model : Torch.nn.model
            PyTorch model trained.

        Returns
        -------
            None
        """
        logging.info('Processing images for epoch {}'.format(epoch_number))

        # Create epoch folder
        epoch_name = 'epoch' + str(epoch_number)
        epoch_folder = os.path.join(self.data_folder, epoch_name)
        os.mkdir(epoch_folder)

        # Create epoch entry in manifest
        self.store['epochs'][epoch_number] = OrderedDict()
        self.store['epochs'][epoch_number]['number'] = epoch_number
        self.store['epochs'][epoch_number]['accuracy'] = epoch_accuracy
        self.store['epochs'][epoch_number]['folder'] = os.path.join('/', epoch_name)
        self.store['epochs'][epoch_number]['datetime'] = datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

        # Collect activations
        activations = self._process(model)

        # Prepare class/general mean
        classes = OrderedDict()
        general = OrderedDict()

        for i in activations:
            item_info = self._get_item_info('single', i)
            index = item_info['class']

            # Add in classes array
            if not index in classes:
                classes[index] = OrderedDict()
                classes[index] = deepcopy(activations[i])
            else:
                for lkey, lval in classes[index]['layers'].items():
                    if 'filters' in lval:
                        for fkey, fval in lval['filters'].items():
                            classes[index]['layers'][lkey]['filters'][fkey] = (
                                    fval + activations[i]['layers'][lkey]['filters'][fkey]
                            )

            # Add in general array
            if not 0 in general:
                general[0] = OrderedDict()
                general[0] = deepcopy(activations[i])
            else:
                for lkey, lval in general[0]['layers'].items():
                    if 'filters' in lval:
                        for fkey, fval in lval['filters'].items():
                            general[0]['layers'][lkey]['filters'][fkey] = (
                                    fval + activations[i]['layers'][lkey]['filters'][fkey]
                            )

        for ckey, cval in classes.items():
            item_info = self._get_item_info('class', ckey)

            for lkey, lval in cval['layers'].items():
                if 'filters' in lval:
                    for fkey, fval in lval['filters'].items():
                        classes[ckey]['layers'][lkey]['filters'][fkey] = (
                                classes[ckey]['layers'][lkey]['filters'][fkey] / item_info['size']
                        )

        for gkey, gval in general.items():
            item_info = self._get_item_info('general', gkey)

            for lkey, lval in gval['layers'].items():
                if 'filters' in lval:
                    for fkey, fval in lval['filters'].items():
                        general[gkey]['layers'][lkey]['filters'][fkey] = (
                                general[gkey]['layers'][lkey]['filters'][fkey] / item_info['size']
                        )

        for i in activations:
            item_info = self._get_item_info('single', i)

            with open(os.path.join(epoch_folder, item_info['key'] + '.json'), 'w') as out:
                json.dump(activations[i], out, indent=2)

        for i in classes:
            item_info = self._get_item_info('class', i)

            with open(os.path.join(epoch_folder, item_info['key'] + '.json'), 'w') as out:
                json.dump(classes[i], out, indent=2)

        for i in general:
            item_info = self._get_item_info('general', i)

            with open(os.path.join(epoch_folder, item_info['key'] + '.json'), 'w') as out:
                json.dump(general[i], out, indent=2)

        self._save()

    def _save(self):
        """
        Write global internal storage on the disk.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
        manifest_path = os.path.join(self.log_folder, 'manifest.json')

        with open(manifest_path, 'w') as out:
            json.dump(self.store, out, indent=2)

    def _get_item_info(self, store_type, index):
        return self.store['items'][store_type][index]

    def _process(self, model):
        images = OrderedDict()

        pbar = tqdm(enumerate(self.dataset), total=self.process_size, ncols=100, leave=False)
        for i, (image, label) in pbar:
            # break policy
            if i >= self.process_size:
                break

            # Prepare data loader to be on CUDA or not
            if not self.no_cuda:
                image = image.cuda()
                label = label.cuda()

            image = Variable(image)
            label = Variable(label)

            input_index = str(i)
            input_class = int(label[0])

            layers = Activation._capture_activations(model, image, self.no_cuda, True)

            # store activation 
            images[input_index] = OrderedDict()
            images[input_index]['layers'] = layers

        return images

    @staticmethod
    def _capture_activations(model, data_input, no_cuda, store_filters=True):
        store = OrderedDict()

        if not no_cuda:
            model = model.module

        for l, layer in enumerate(model.children()):
            data_input = layer(data_input)
            layer_dim = data_input.dim()

            layer_name = str(l + 1)
            store[layer_name] = OrderedDict()

            if not store_filters:
                store[layer_name]['type'] = str(layer)  # TODO: store something cleaner
                store[layer_name]['dim'] = layer_dim
                store[layer_name]['size'] = data_input.size()[1]

            if store_filters:
                numpy_filter = np.array([])
                store[layer_name]['filters'] = OrderedDict()

                if layer_dim == 4:
                    # dimension 1 is for the mini-batch
                    for f in range(0, data_input.size()[1]):
                        # mean
                        fa = data_input[0, f].data.permute(0, 1).cpu().numpy()
                        numpy_filter = np.append(numpy_filter, np.mean(fa))
                elif layer_dim == 2:
                    for f in range(0, data_input.size()[1]):
                        numpy_filter = np.append(numpy_filter, float(data_input[0, f].data))
                else:
                    # is that even possible? 
                    numpy_filter = np.append(numpy_filter, 0)

                # normalize data along the layer
                if numpy_filter.min() < 0:
                    numpy_filter -= numpy_filter.min()
                numpy_filter *= 1 / numpy_filter.max()

                # store
                for index, value in np.ndenumerate(numpy_filter):
                    store[layer_name]['filters'][str(index[0] + 1)] = float(value)

        return store
