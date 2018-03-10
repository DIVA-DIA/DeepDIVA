"""
This file is the template for the boilerplate of train/test of a triplet network.
This code has initially been adapted to our purposes from:
        PyTorch training code for TFeat shallow convolutional patch descriptor:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf

        The code reproduces *exactly* it's lua anf TF version:
        https://github.com/vbalnt/tfeat

        2017 Edgar Riba

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Michele Alberti, Vinaychandran Pondenkandath
"""
from __future__ import print_function

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from tqdm import tqdm

from template.runner.triplet.eval_metrics import ErrorRateAt95Recall
# DeepDIVA
# Delegated
from template.setup import set_up_model


# Utils


#######################################################################################################################
class Triplet:
    @staticmethod
    def single_run(writer, log_dir, model_name, epochs, lr, decay_lr, **kwargs):

        test_loader, train_loader = Triplet.setup_dataloaders(**kwargs)

        # Setting up model, optimizer, criterion
        # TODO this has to be replaced with a custom ting for the triplet most probably
        model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=2,
                                                                            model_name=model_name,
                                                                            lr=lr,
                                                                            train_loader=train_loader,
                                                                            **kwargs)

        # initialize weights
        # TODO check is this is done anyway by default?
        model.apply(Triplet.weights_init)

        optimizer = torch.optim.__dict__[kwargs['optimizer_name']](model.parameters(), lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, dampening=0.9,
                                    weight_decay=1e-4)

        for epoch in range(start_epoch, epochs):
            Triplet.train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
                          writer=writer, epoch=epoch, lr=lr, **kwargs)
            Triplet.test(test_loader, model, criterion, writer, epoch, **kwargs)

    @staticmethod
    def setup_dataloaders(dataset_folder, n_triplets, batch_size, workers, **kwargs):
        cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
        np_reshape = lambda x: np.reshape(x, (32, 32, 1))

        #kwargs = {'num_workers': 2, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            Triplet.TripletPhotoTour(train=True,
                                     n_triplets=n_triplets,
                                     root=dataset_folder,
                                     name='yosemite',
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.Lambda(cv2_scale),
                                         transforms.Lambda(np_reshape),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.48544601108437,), (0.18649942105166,))
                                     ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            Triplet.TripletPhotoTour(train=False,
                                     root=dataset_folder,
                                     name='liberty',
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.Lambda(cv2_scale),
                                         transforms.Lambda(np_reshape),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.48544601108437,), (0.18649942105166,))
                                     ])),
            batch_size=1000,
            shuffle=False,
            num_workers=workers,
            pin_memory=True)
        return test_loader, train_loader

    class Logger(object):
        def __init__(self, log_dir):
            # clean previous logged data under the same directory name
            self._remove(log_dir)

            # configure the project
            configure(log_dir)

            self.global_step = 0

        def log_value(self, name, value):
            log_value(name, value, self.global_step)
            return self

        def step(self):
            self.global_step += 1

        @staticmethod
        def _remove(path):
            """ param <path> could either be relative or absolute. """
            if os.path.isfile(path):
                os.remove(path)  # remove the file
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)  # remove dir and all contains

    class TripletPhotoTour(dset.PhotoTour):
        """From the PhotoTour Dataset it generates triplet samples
        note: a triplet is composed by a pair of matching images and one of
        different class.
        """

        def __init__(self, train=True, transform=None, n_triplets=10000, *arg, **kwargs):
            super(Triplet.TripletPhotoTour, self).__init__(*arg, **kwargs)
            self.transform = transform

            self.train = train
            self.n_triplets = n_triplets

            if self.train:
                print('Generating {} triplets'.format(self.n_triplets))
                self.triplets = self.generate_triplets(self.labels, self.n_triplets)

        @staticmethod
        def generate_triplets(labels, num_triplets):
            def create_indices(_labels):
                inds = dict()
                for idx, ind in enumerate(_labels):
                    if ind not in inds:
                        inds[ind] = []
                    inds[ind].append(idx)
                return inds

            triplets = []
            indices = create_indices(labels)
            unique_labels = np.unique(labels.numpy())
            n_classes = unique_labels.shape[0]

            for x in tqdm(range(num_triplets)):
                c1 = np.random.randint(0, n_classes - 1)
                c2 = np.random.randint(0, n_classes - 1)
                while c1 == c2:
                    c2 = np.random.randint(0, n_classes - 1)
                if len(indices[c1]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices[c1]) - 1)
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices[c1]) - 1)
                n3 = np.random.randint(0, len(indices[c2]) - 1)

                triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
            return torch.LongTensor(np.array(triplets))

        def __getitem__(self, index):
            def transform_img(img):
                if self.transform is not None:
                    img = self.transform(img.numpy())
                return img

            if not self.train:
                m = self.matches[index]
                img1 = transform_img(self.data[m[0]])
                img2 = transform_img(self.data[m[1]])
                return img1, img2, m[2]

            t = self.triplets[index]
            a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

            # transform images if required
            img_a = transform_img(a)
            img_p = transform_img(p)
            img_n = transform_img(n)
            return img_a, img_p, img_n

        def __len__(self):
            if self.train:
                return self.triplets.size(0)
            else:
                return self.matches.size(0)


    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight.data, gain=math.sqrt(2.0))
            nn.init.constant(m.bias.data, 0.1)

    def train(train_loader, model, criterion, optimizer, writer, epoch, no_cuda, margin, anchorswap, lr,
              log_interval=25, **kwargs):
        # switch to train mode
        model.train()

        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data_a, data_p, data_n) in pbar:

            if not no_cuda:
                data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()

            data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)

            # compute output
            out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
            loss = F.triplet_margin_loss(out_p, out_a, out_n, margin=margin, swap=anchorswap)
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the optimizer learning rate
            Triplet.adjust_learning_rate(optimizer, lr)

            # log loss value
            # # create logger
            # logger = Triplet.Logger(log_dir)
            # logger.log_value('loss', loss.data[0]).step()

            if batch_idx % log_interval == 0:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_a), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.data[0]))

        # do checkpointing
                # torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                #            '{}/checkpoint_{}.pth'.format(args.log_dir, epoch))

    def test(test_loader, model, criterion, writer, epoch, no_cuda, log_interval=25, **kwargs):
        # switch to evaluate mode
        model.eval()

        labels, distances = [], []

        pbar = tqdm(enumerate(test_loader))
        for batch_idx, (data_a, data_p, label) in pbar:
            if not no_cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()

            data_a, data_p, label = Variable(data_a, volatile=True), Variable(data_p, volatile=True), Variable(label)

            # compute output
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

            if batch_idx % log_interval == 0:
                pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader)))

        # measure accuracy (FPR95)
        num_tests = test_loader.dataset.matches.size(0)
        labels = np.vstack(labels).reshape(num_tests)
        distances = np.vstack(distances).reshape(num_tests)

        fpr95 = ErrorRateAt95Recall(labels, distances)
        print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

        # Triplet.logger.log_value('fpr95', fpr95)

    def adjust_learning_rate(optimizer, lr, lr_decay=1e-6):
        """Updates the learning rate given the learning rate decay.
        The routine has been implemented according to the original Lua SGD optimizer
        """
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1

            group['lr'] = lr / (1 + group['step'] * lr_decay)
