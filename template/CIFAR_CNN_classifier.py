import argparse
import os
import time
import logging
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn  as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision import models
from tensorboardX import SummaryWriter

from dataset import CIFAR10, CIFAR100
from model import CNN_basic
from util.misc import AverageMeter, accuracy

# Argument Parser

# Training Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Template for training CNN on CIFAR')

# General Options
parser.add_argument('--experiment-name',
                    help='provide a meaningful and descriptive name to this run',
                    default=None, type=str)

# Data Options
parser.add_argument('--dataset',
                    help='one of {CIFAR10, CIFAR100}', default='CIFAR10')
parser.add_argument('--log-dir',
                    help='where to save logs', default='./data/')

# Training Options
parser.add_argument('--lr',
                    help='learning rate to be used for training', type=float, default=0.0001)
parser.add_argument('--optimizer',
                    help='optimizer to be used for training. {Adam, SGD}', default='Adam')
parser.add_argument('--batch-size',
                    help='input batch size for training', type=int, default=64)
parser.add_argument('--test-batch-size',
                    help='input batch size for testing', type=int, default=64)
parser.add_argument('--epochs',
                    help='how many epochs to train', type=int, default=30)
parser.add_argument('--resume',
                    help='path to latest checkpoint', default=None, type=str)

# System Options
parser.add_argument('--gpu-id',
                    default=None, help='which GPUs to use for training (use all by default)')
parser.add_argument('--no-cuda',
                    default=False, action='store_true', help='run on CPU')
parser.add_argument('--seed',
                    default=None, help='random seed')
parser.add_argument('--log-interval',
                    default=10, type=int, help='print loss/accuracy every N batches')
parser.add_argument('-j', '--workers',
                    default=4, type=int, help='workers used for train/val loaders')
args = parser.parse_args()

# Set visible GPUs
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Setup Logging
basename = args.log_dir
experiment_name = args.experiment_name
log_folder = os.path.join(basename, experiment_name, '{}'.format(time.strftime(('%y-%m-%d-%Hh-%Mm-%Ss'))))
logfile = 'logs.txt'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
logging.basicConfig(format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
                    filename=os.path.join(log_folder, logfile),
                    level=logging.INFO)
logging.info('Set up logging. Log file: {}'.format(os.path.join(log_folder, logfile)))


# Save args to logs_folder
logging.info('Arguments saved to: {}'.format(os.path.join(log_folder, 'args.txt')))
with open(os.path.join(log_folder, 'args.txt'), 'w') as f:
    f.write(json.dumps(vars(args)))

# Define Tensorboard SummaryWriter
logging.info('Initialize Tensorboard SummaryWriter')
writer = SummaryWriter(log_dir=log_folder)




def main():

    logging.info('Initalizing dataset {}'.format(args.dataset))
    if args.dataset == 'CIFAR10':
        train_ds = CIFAR10(root='.data/',
                           train=True,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

        test_ds = CIFAR10(root='.data/',
                          train=False,
                          download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))
        num_outputs = 10
    else:
        train_ds = CIFAR100(root='.data/',
                           train=True,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

        test_ds = CIFAR100(root='.data/',
                          train=False,
                          download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))
        num_outputs = 100

    logging.info('Set up dataloaders')
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              pin_memory=True)

    logging.info('Initialize model')
    model = CNN_basic.CNN_Basic(num_outputs)
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    if not args.no_cuda:
        logging.info('Transfer model to GPU')
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    logging.info('Begin training')
    for i in range(args.epochs):
        train(train_loader, model, criterion, optimizer, i)
        validate(test_loader, model, criterion, i)

    logging.info('Training completed')
    writer.close()



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if not args.no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        writer.add_scalar('train/loss', loss.data[0], epoch*len(train_loader) + i)
        writer.add_scalar('train/accuracy', prec1.cpu().numpy(), epoch*len(train_loader) + i)

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        writer.add_scalar('test/loss', loss.data[0], epoch * len(val_loader) + i)
        writer.add_scalar('test/accuracy', prec1.cpu().numpy(), epoch * len(val_loader) + i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return


if __name__ == "__main__":


    # Set up logging to console
    fmtr = logging.Formatter(fmt='%(funcName)s %(levelname)s: %(message)s')
    stderr_handler = logging.StreamHandler()
    stderr_handler.formatter = fmtr
    logging.getLogger().addHandler(stderr_handler)
    logging.info('Printing activity to the console')

    main()