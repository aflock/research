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
'''


