#!/usr/bin/env python
# encoding: utf-8


'''#{{{
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
                right[k] += 1


Nearest Neighbors(sample, k):
    for p in training set:
        d = euclidean_dist(sample, p)
        if d < kth best d:
            update kth best
    return kbest
'''#}}}

#imports
import sys, os
import numpy as np
from random import shuffle
import pickle

#load samples
# TODO: fill this in plz VV (01/29/12, 21:42, AFlock)
DATA_FILE = open("./data.p", "rb")
samples = pickle.load(DATA_FILE)
shuffle(samples)
num_s = len(samples)


training_set = samples[:num_s/2]
remainder = samples[num_s/2:]
test_set = remainder[:len(remainder)/6]

def main():
    """runs nearest neighbors """

    pass

def nearest_neighbors(sample, k):
    """Finds k nearest neighbors based on Euclidean distance

    :sample: the sample
    :k: how many neighbors to find
    :returns: the neighbors themselves
        (since we may want to show them, we will do the voting in main)
    """
    scores = {}
    for t_sample in training_set:
        dist = np.linalg.norm(sample[0]-t_sample[0])
        scores[t_sample] = dist

    list =  sorted(training_set, key=scores.__getitem__).reverse()
    list.reverse()
    return list[:k]
