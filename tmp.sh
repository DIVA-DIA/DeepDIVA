#!/usr/bin/env bash

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name no_seed1 --epochs 3 --model-name resnet18

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name no_seed2 --epochs 3 --model-name resnet18

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name no_seed3 --epochs 3 --model-name resnet18 --gpu-id 0



python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name seed1 --epochs 3 --model-name resnet18 --seed 42

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name seed1 --epochs 3 --model-name resnet18 --seed 42

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name seed2 --epochs 3 --model-name resnet18 --gpu-id 0 --seed 42

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name seed2 --epochs 3 --model-name resnet18 --gpu-id 0 --seed 42

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name seed3 --epochs 3 --model-name resnet18 --no-cuda --seed 42

python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --experiment-name seed3 --epochs 3 --model-name resnet18 --no-cuda --seed 42