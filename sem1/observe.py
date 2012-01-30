#!/usr/bin/python
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import sys, os
'''
File: observe.py
Author: AFLOCK
Description: view the images resulting from last nights tests.
'''

def main(number, option):
    """viewj"""
    pics = [
            "diffedpic_j8m862g8s_raw.fits.p",
            "diffedpic_j8m862gbq_raw.fits.p",
            "diffedpic_j8m862goq_raw.fits.p",
            "diffedpic_j8m862gtq_raw.fits.p"
            ]

    mins = [
            "shift_resultsset1_j8m862g8s_raw.fits.p",
            "shift_resultsset1_j8m862gbq_raw.fits.p",
            "shift_resultsset1_j8m862goq_raw.fits.p",
            "shift_resultsset1_j8m862gtq_raw.fits.p"
            ]

    if option == "pics":
        theList = pics
    else:
        theList = mins
    index = int(number)
    f1 = pickle.load(open(theList[index], "rb"))

    if option == "pics":
        median = np.median(f1)
        #plt.imshow(f1, cmap = cm.gray, vmin = median-70, vmax = median+70)
        #np.log(f1)
        #hist = plt.hist(f1.flatten(), bins = 250,   log=True)
        #range = (-4000,4000),
        pt1 = 1300
        pt2 = 1600
        img = f1[pt1:pt2, pt1:pt2]
        #plt.imshow(img, cmap =  cm.gray, vmin = median-70, vmax = median+70)
        plt.imshow(f1, cmap =  cm.gray, vmin = median-70, vmax = median+70)
    else:
        plt.imshow(f1, cmap = cm.hot)
        plt.colorbar()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
