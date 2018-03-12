# Utils
from __future__ import print_function

# Torch
import torch
import torch.nn.init
import torchvision.transforms as transforms

# DeepDIVA
from datasets.Triplet_PhotoTour import TripletPhotoTour


def setup_dataloaders(model_expected_input_size, dataset_folder, n_triplets, batch_size, workers, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         n_triplets=n_triplets,
                         root=dataset_folder,
                         name='yosemite',
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(model_expected_input_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.48544601108437,), (0.18649942105166,))
                         ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=False,
                         root=dataset_folder,
                         name='liberty',
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(model_expected_input_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.48544601108437,), (0.18649942105166,))
                         ])),
        batch_size=1000,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)
    return test_loader, train_loader
