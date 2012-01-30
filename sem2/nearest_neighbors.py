#!/usr/bin/env python
# encoding: utf-8

#imports
import sys, os
import numpy as np
import pickle

# TODO: fill this in plz VV (01/29/12, 21:42, AFlock)
DATA_FILE = "./data.p"
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
                right[k] += 1


Nearest Neighbors(sample, k):
    for p in training set:
        d = euclidean_dist(sample, p)
        if d < kth best d:
            update kth best
    return kbest
'''
def main():
    """runs nearest neighbors """
    #get data
    samples = pickle.load("DATA_FILE")

    pass

def nearest_neighbors(sample, k):
    """Finds k nearest neighbors based on Euclidean distance

    :sample: the sample
    :k: how many neighbors to find
    :returns: the neighbors themselves
        (since we may want to show them, we will do the voting in main)
    """

    pass
