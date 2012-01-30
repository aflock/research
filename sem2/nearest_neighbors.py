#!/usr/bin/env python
# encoding: utf-8

'''
File: nearest_neighbors.py
Author: AFlock
Description: Nearest neighbors algorithm for classification of Cosmic Ray pixels in HST images
            Assumes that samples are arrayed as a list of tuples as follows:

            samples  = [
                (<tuple or list of pixel values for the sample> , <1/0> ),
                (<tuple or list of pixel values for the sample> , <1/0> ),
                (<tuple or list of pixel values for the sample> , <1/0> ),
                (<tuple or list of pixel values for the sample> , <1/0> ),
                (<tuple or list of pixel values for the sample> , <1/0> ),
                (<tuple or list of pixel values for the sample> , <1/0> ),
                (<tuple or list of pixel values for the sample> , <1/0> ),
            ]

Pseudo code:
    training set = 1/2
    test set = 5/12
    validation set = 1/12



    for k in [1,3,5,7,9]:
        for sample, answer in VALIDATION set:
            Get k nearest neighbors from training set = NN
            guess = NN.vote()
            if guess == answer :

'''


