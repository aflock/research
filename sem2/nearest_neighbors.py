#!/usr/bin/env python
# encoding: utf-8
'''
File: nearest_neighbors.py
Author: AFlock
Description: Nearest neighbors algorithm for classification of Cosmic Ray pixels in HST images
'''

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
validation_set = remainder[:len(remainder)/6]
test_set = remainder[len(remainder)/6:]


def main():
    """runs nearest neighbors """
    k_scores = {}
    for k in [1,3,5,7,9]:
        k_scores[k] = 0
        for v_sample in validation_set:
            NN = nearest_neighbors(v_sample, k)
            votes_for_CR = sum([1 for x in NN if NN[1] is 1])
            if votes_for_CR > k/2:
                cr_consensus = 1
            else:
                cr_consensus = 0

            true_answer = v_sample[1]
            if true_answer == cr_consensus:
                k_scores[k] +=1
    print k_scores


def nearest_neighbors(sample, k):
    """Finds k nearest neighbors based on Euclidean distance
    :returns: the neighbors themselves
        (since we may want to show them, we will do the voting in main)
    """
    scores = {}
    for t_sample in training_set:
        # TODO: might this be done faster? (01/30/12, 11:42, AFlock)
        dist = np.linalg.norm(sample[0]-t_sample[0])
        scores[t_sample] = dist

    list =  sorted(training_set, key=scores.__getitem__)
    list.reverse()
    return list[:k]


if __name__ == '__main__':
    main()
