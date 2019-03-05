# PyTorch Implementation of Non-Local Neural Network

This repository contains my implementation of non-local neural netowrks (CVPR 2018).

The experiment was run on CIFAR-10 dataset for the sake of ensuring that the code runs without error.

## Details
The original paper used ResNet-50 as its backbone structure for conducting experiment on video datasets such as Kinetics, Charades.

As an inital study, I adopted ResNet-56 strucutre for CIFAR-10 dataset which is a 2D classification.

The four different pairwise functions discussed in the paper are implemented accordingly.

## TO DO
- [ ] Compare the result of baseline model and that of non-local model for CIFAR-10
- [ ] Prepare video dataset (e.g. UCF-101, HMDB-51)
- [ ] Modify the model code to adapt to spatiotemporal settings
- [ ] Run test on these video datasets
- [ ] Run test on image segmentation dataset (e.g. COCO)

## Reference
This repo is an adaptation from several other exisitng works.
- https://github.com/akamaster/pytorch_resnet_cifar10
- https://github.com/kuangliu/pytorch-cifar
- https://github.com/AlexHex7/Non-local_pytorch
