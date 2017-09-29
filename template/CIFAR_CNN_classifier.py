from dataset import CIFAR10, CIFAR100
from model import CNN_basic

import torch
import torch.nn  as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


def train(train_loader, model, criterion, optimizer):
    for i, (input, target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print('{} {}'.format(i, loss.data[0]))


train_ds = CIFAR100(root='.data/',
                   train=True,
                   download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

test_ds = CIFAR100(root='.data/',
                  train=False,
                  download=True,
                  transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=64,
                                           num_workers=2,
                                           pin_memory=True,
                                           )


test_loader = torch.utils.data.DataLoader(test_ds,
                                           batch_size=64,
                                           num_workers=2,
                                           pin_memory=True,
                                           )


model = CNN_basic.CNN_Basic(100).cuda()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
criterion = nn.CrossEntropyLoss().cuda()
cudnn.benchmark = True

for i in range(100):
    train(train_loader, model, criterion, optimizer)





