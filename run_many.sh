#!/usr/bin/env bash

#CIFAR
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10 --gpu-id 0 --experiment-name CIFAR_CNN_basic_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10 --model-name babyresnet18 --experiment-name CIFAR_babyresnet18_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10 --model-name resnet18 --experiment-name CIFAR_resnet18_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10 --model-name densenet121 --experiment-name CIFAR_densenet121_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10 --model-name vgg16 --experiment-name CIFAR_vgg16_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10 --model-name alexnet --experiment-name CIFAR_alexnet_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20

#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper --gpu-id 0 --experiment-name CIFAR_CNN_basic_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper --model-name babyresnet18 --experiment-name CIFAR_babyresnet18_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper --model-name resnet18 --experiment-name CIFAR_resnet18_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper --model-name densenet121 --experiment-name CIFAR_densenet121_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper --model-name vgg16 --experiment-name CIFAR_vgg16_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper --model-name alexnet --experiment-name CIFAR_alexnet_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20


#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper-moving --gpu-id 0 --experiment-name CIFAR_CNN_basic_tamper_moving_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper-moving --model-name babyresnet18 --experiment-name CIFAR_babyresnet18_tamper_moving_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/CIFAR10-tamper-moving --model-name resnet18 --experiment-name CIFAR_resnet18_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20



#SVHN
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN --gpu-id 0 --experiment-name SVHN_CNN_basic_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN --model-name babyresnet18 --experiment-name SVHN_babyresnet18_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN --model-name resnet18 --experiment-name SVHN_resnet18_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN --model-name densenet121 --experiment-name SVHN_densenet121_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN --model-name vgg16 --experiment-name SVHN_vgg16_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN --model-name alexnet --experiment-name SVHN_alexnet_baseline -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper --gpu-id 0 --experiment-name SVHN_CNN_basic_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper --model-name babyresnet18 --experiment-name SVHN_babyresnet18_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper --model-name resnet18 --experiment-name SVHN_resnet18_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper --model-name densenet121 --experiment-name SVHN_densenet121_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper --model-name vgg16 --experiment-name SVHN_vgg16_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper --model-name alexnet --experiment-name SVHN_alexnet_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20


#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper-moving --gpu-id 0 --experiment-name SVHN_CNN_basic_tamper_moving_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper-moving --model-name babyresnet18 --experiment-name SVHN_babyresnet18_tamper_moving_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
#python template/RunMe.py --dataset-folder /scratch/tamper_experiments/SVHN-tamper --model-name resnet18 --experiment-name SVHN_resnet18_tamper_1px -j 16 --ignoregit --seed 42 --lr 0.01 --momentum 0.9 --epochs 20
