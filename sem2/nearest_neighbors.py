#!/usr/bin/env python
# encoding: utf-8
'''
File: nearest_neighbors.py
Author: AFlock
Description: Nearest neighbors algorithm for classification of Cosmic Ray pixels in HST images
'''

#imports
import numpy as np
import pickle
from random   import shuffle
from optparse import OptionParser

#load samples
data_dir = "/misc/vlgscratch1/FergusGroup/abf277/hst"


training_set = []
def main(options, args):
    print "options", options
    sd = options.sd
    neg_samples = pickle.load(open("%s/samples/negative_%s.p" % (data_dir, sd),"rb"))
    pos_samples = pickle.load(open("%s/samples/positive_%s.p" % (data_dir, sd),"rb"))
    samples = neg_samples + pos_samples
    #sanitize for incorrectly shaped samples
    for s in samples:
        if s['x'].size is not 81:
            print s['x'].size
            print "removing sample : \n", s
            samples.remove(s)
    shuffle(samples)
    num_s = len(samples)
    global training_set
    training_set = samples[:num_s/2]
    remainder = samples[num_s/2:]
    validation_set = remainder[:len(remainder)/6]
    test_set = remainder[len(remainder)/6:]
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

    parser = OptionParser()
    parser.add_option("-d", "--sample_date", dest="sd",
            help ="which date of samples we want to use", default="2012-04-01")
    (options, args) = parser.parse_args()
    print "start"
    main(options, args)

