#!/usr/bin/python
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
File: shift_calculation.py
Author: Aflock
Description: calculate error with different methods, rates of eta.
'''

def errorNaive(shifted, comparison):
    diff = shifted-comparison
    diff = np.square(diff)
    diff = np.sum(diff)

    return diff

def errorWise(shifted, comparison, mad, epsilon):

    diff = shifted-comparison
    diff = np.square(diff)
    diff = np.sum(diff)

    epmad = (epsilon* np.square(mad))
    print epmad
    diff = diff/(diff+ epmad)

    return diff

def main(epsilon):
    files = ['j8m862gtq_raw.fits','j8m81cd8q_raw.fits']
    f1 = files[0]
    f2 = files[1]
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    filename = data_dir + f1
    filename2 = data_dir + f2
    raw_data = pft.open(filename)
    raw_data2 = pft.open(filename2)
    sci_data = raw_data[1].data
    sci_data2 = raw_data2[1].data
    pt1 = 0
    pt2 = 2000
    image1 = sci_data[pt1:pt2, pt1:pt2]    #sci_data
    image2 = sci_data2[pt1:pt2, pt1:pt2]   #sci_data2



    #calculate mad
    diff = image1 - image2
    median = np.median(diff)
    absdif = np.absolute(diff)
    mad = np.median(absdif)


    print image1[0].size
    print image1[1].size

    #interpolate across image 1
    time1 = dt.datetime.now()
    interpoint = intpl.RectBivariateSpline(np.arange(image1[0].size),
            np.arange(image1[1].size), image1, kx = 5, ky=5)


    count = 0
    gi = -1
    gj = -1
    result = np.zeros((100,100))
    minDiff = np.inf
    minShift = [0,0]

    for i in np.arange(-0.5, .5, 0.01):
        gi+=1
        gj=-1
        for j in np.arange(-0.5, .5, 0.01):
            count += 1
            time1 = dt.datetime.now()
            gj+=1
            target = interpoint.__call__(np.arange(image1[0].size)+i, np.arange(image1[0].size)+j)
            diff = errorWise(target, image2, mad, epsilon)

            if diff < minDiff:
                minDiff = diff
                minShift[0] = i
                minShift[1] = j

            result[gi,gj] = diff
            time2 = dt.datetime.now()
            print "time for interation ", count, " : ", time2-time1
            print "diff is ", diff, "for x change: ", i, " :: ychange : ", j
            print "^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^"
            print "^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^"

    print "Minimum point: ", np.min(result)
    print "shift : " , minShift
    print "Maximum point: ", np.max(result)

    minImg = interpoint.__call__(np.arange(image1[0].size)+minShift[0], np.arange(image1[0].size)+minShift[1])
    sub = (minImg - image2)

    pickle.dump(result, open("%sshift_results_%s_%s.p" % (data_dir, epsilon, errorMethod), "wb"))
    pickle.dump(sub, open("%sdiffedpic_%s_%s.p" % (data_dir, epsilon, errorMethod), "wb"))

    median = np.median(minImg- image2)

    #plt.imshow(result, cmap = cm.hot)
    #plt.imshow(sub, cmap = cm.gray , vmin =median-70, vmax =median+70)
    #plt.colorbar()

    return minShift

if __name__ == '__main__':
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    shifts = []
    epsilons =  [0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100, 600, 1000]

    for ep in epsilons:
        mins = main(ep)
        shifts.append(mins)
        print "done with epsilon ", ep

    pickle.dump(shifts, open("%s/shiftacrossepsilon.p", "wb" % (data_dir)))
