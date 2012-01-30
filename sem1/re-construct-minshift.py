import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.transforms as trsnfm
import numpy as np
import scipy.interpolate as intpl
import pyfits as pft
import pickle
import sys, os
import datetime as dt

'''
File: re-construct-minshift.py
Author: AF
Description: find the minshift for all epsilons
'''
def main():
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    filename_filter = "shift_results"
    picks = []
    for filename in os.listdir(data_dir) :
        if filename_filter in filename:
            picks.append(filename)

    data = {}
    for f in picks:
        l = f.split("_")
        ep = l[2]
        data[ep] = pickle.load(open(data_dir+f, "rb"))


    #now. what is the coordinate of the min?
    #these are all steps of .1 from [-.5, .5]
    shifts = {}
    for d, arr in data.iteritems():
        shifts[d] = np.unravel_index(arr.argmin(), (100,100))

    print shifts
if __name__ == '__main__':
    main()
