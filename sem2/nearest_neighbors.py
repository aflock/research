#!/usr/bin/env python
# encoding: utf-8
'''
File: nearest_neighbors.py
Author: AFlock
Description: Nearest neighbors algorithm for classification of Cosmic Ray pixels in HST images
'''

#imports
import numpy as np
from random import shuffle
import pickle

#load samples
data_dir = "/misc/vlgscratch1/FergusGroup/abf277/hst"
neg_samples = pickle.load(open("%s/samples/negative.p" % data_dir,"rb"))
pos_samples = pickle.load(open("%s/samples/positive.p" % data_dir,"rb"))
samples = neg_samples + pos_samples
shuffle(samples)
num_s = len(samples)
training_set = samples[:num_s/2]
remainder = samples[num_s/2:]
validation_set = remainder[:len(remainder)/6]
test_set = remainder[len(remainder)/6:]


def main():
    """runs nearest neighbors """
    k_scores = {}
    print "length of validation set : %s" % len(validation_set)
    for k in [1,3,5,7,9]:
        print "k is %s " % k
        k_scores[k] = 0
        for v_sample in validation_set:
            NN = nearest_neighbors(v_sample, k)
            votes_for_CR = sum([1 for x in NN if x['y'] is 1])
            if votes_for_CR > k/2:
                cr_consensus = 1
            else:
                cr_consensus = 0

            true_answer = v_sample['y']
            if true_answer == cr_consensus:
                k_scores[k] +=1
    print k_scores


def nearest_neighbors(sample, k):
    """Finds k nearest neighbors based on Euclidean distance
    :returns: the neighbors themselves
        (since we may want to show them, we will do the voting in main)
    """
    scores = {}
    for index, t_sample in enumerate(training_set):
        #print "t sample is:" , t_sample
        # TODO: might this be done faster? (01/30/12, 11:42, AFlock)
        dist = np.linalg.norm(sample['x']-t_sample['x'])
        scores[index] = dist

    list =  sorted(range(len(training_set)), key=scores.__getitem__)
    list.reverse()
    #print [training_set[index] for index in list[:k]]
    return [training_set[index] for index in list[:k]]


if __name__ == '__main__':
    #DATA_FILE = open("./data.p", "rb")
    #samples = pickle.load(DATA_FILE)
    samples = [#{{{
            (np.array([1,2,3,1,5,6,0,1,204,6,23,6,2,1,4,6,78,2,1,12,4,5,7,3,1,21,344,2,1,2,3,45,6,67]), 1),
            (np.array([1,2,3,1,24,6,0,1,2015,6,23,6,2,1,4,6,78,2,1,12,4,5,7,3,1,21,34,2,1,2,3,5,6,67]), 0),
            (np.array([1,5,3,-1,5,9,0,2,5,31,253,6,2,169,4,6,78,2,1,12,4,5,7,3,1,21,34,2,1,24,3,-8,6,67]), 1),
            (np.array([1,2,3,1,5,-1101,10,1,156,6,23,6,7,881,37,6,78,-8,1,12,4,5,7,3,1,21,34,2,1,2,-24,5,6,67]), 0),
            (np.array([1,2,3,23,5,6,0,1,218,6,23,6,2,1,4,6,78,2,1,12,4,5,27,3,1,21,34,2,1,-94,17,5,6,67]), 1),
            (np.array([11,2,31,1,5,16,0,1,2063,6,20,6,4,4,4,6,78,2,1,12,4,5,7,17,1,21,34,2,1,2,3,5,17,42]), 0),
            (np.array([1,2,19,1,5,24,1,1,2054,6,93,6,2,1,-6,6,78,2,1,12,4,55,7,6,1,21,34,2,14,2,31,5,6,98]), 1),
            (np.array([1,28,3,-15,5,6,30,109,404,6,23,6,2,1,4,6,78,2,1,12,4,42,8,3,11,2,34,2,17,3,-19,-14,6,67]), 0),
            (np.array([1,2,3,1,5,6,0,1,204,6,23,6,2,1,4,6,78,2,1,12,4,5,7,3,1,21,34,2,1,2,3,5,6,67]), 1),
            (np.array([1,2,24,1,-7,6,0,1,204,6,23,6,2,1,4,6,78,2,1,12,4,5,75,3,516,21,34,2,1,2,3,16,-15,67]), 0),
            (np.array([1,2,8,1,-10,14,0,1,143,6,23,6,19,1,4,6,78,11,1,124,4,16,7,3,41,22,34,2,1,-9,3,5,6,51]), 1),
            (np.array([1,2,3,17,5,6,0,1,204,6,23,6,2,1,4,16,78,2,1,12,4,5,7,3,51,21,34,2,1,2,3,-9,6,67]), 0),
            (np.array([1,-7,3,1,-16,6,20,1,204,6,23,6,2,1,4,6,78,2,1,12,4,5,75,3,-10,21,34,2,1,2,-17,5,6,67]), 0),
            (np.array([42,2,3,1,-10,14,0,1,143,6,23,6,19,1,4,6,78,11,1,12,4,16,7,3,1,22,34,2,1,-9,3,5,6,51]), 1),
            (np.array([1,2,109,1,-25,414,0,1,1434,6,23,6,19,1,4,6,78,11,1,-24,4,16,7,3,1,22,34,2,1,-9,3,5,6,51]), 1),
            (np.array([1,3,4,1,-10,14,0,1,205,6,3323,6,19,1,4,6,78,11,26,412,4,156,7,43,1,22,344,2,1,-9,3,5,6,51]), 1),
            (np.array([4,2,63,1,-40,14,0,1,170,6,23,66,619,1,4,6,78,11,-14,12,4,16,7,3,1,22,34,2,1,-9,3,5,6,51]), 1),
            (np.array([-9,62,3,1,-160,14,0,1,143,6,23,6,19,1,4,71,78,610,1,12,-42,166,7,3,1,22,34,2,1,-9,3,5,6,51]), 0),
            (np.array([16,2,3,1,-10,164,-1,-16,108,0,23,6,19,1,4,28,78,11,16,12,16,16,467,3,1,22,34,2,1,-9,3,5,6,51]), 1),
            (np.array([17,7,3,1,-10,146,0,1,143,71,23,-35,19,1,4,66,78,11,16,12,55,16,7,3,1,22,34,2,1,-9,3,5,6,51]), 0),
            (np.array([1,2,19,1,-25,14,0,1,143,6,23,6,19,15,4,6,78,11,1,12,4,16,7,3,1,22,34,2,1,-9,3,5,6,591]), 1),
            (np.array([156,2,19,6,-25,14,0,1,143,56,23,6,19,51,4,6,78,45,1,12,4,16,36,3,1,22,344,42,1,-9,3,5,6,51]), 0),
            (np.array([1,2,19,15,-15,14,0,1,1447,6,23,6,19,1,4,6,78,45,1,12,4,16,7,3,1,22,34,2,1,-9,3,5,6,51]), 1),
            (np.array([51,2,19,1,-254,14,0,1,143,24,28,6,19,1,4,6,78,1144,1,12,4,16,7494,3,1,22,34,2,1,-99,3,5,6,51]), 0),
            (np.array([1,2,19,1,-24,14,0,1,143,6,23,16,19,1,44,6,78,151,1,12,4,16,7,43,1,22,344,2,1,-9,3,95,6,51]), 1),
            (np.array([1,2,19,1,-2,14,0,1,143,46,23,-34,19,1,544,6,785,11,1,12,-8,16,7,3,91,22,34,2,1,-9,3,5,6,51]), 1),
            (np.array([1,2,19,51,-25,145,0,1,143,65,28,6,19,51,4,6,78,1551,1,12,4,16,7,3,1,22,34,2,1,-9,3,95,6,51]), 1)
            ]#}}}

    print "start"
    main()

