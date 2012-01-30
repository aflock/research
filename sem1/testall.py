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

import isolate_cr

def realInterp():
    the_pictures = [['j8m862gtq_raw.fits','j8m81cd8q_raw.fits']]
    for set in the_pictures:
        f1 = set[0]
        f2 = set[1]
        data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
        filename = data_dir + f1
        filename2 = data_dir + f2
        raw_data = pft.open(filename)
        raw_data2 = pft.open(filename2)
        sci_data = raw_data[1].data
        sci_data2 = raw_data2[1].data

        image1 = sci_data
        image2 = sci_data2

        diff = image1 - image2
        median = np.median(diff)
        print median
        print image1[0].size
        print (image1.size/ image1[1].size)
        #interpolate across image 1
        time1 = dt.datetime.now()

        #interpoint = intpl.interp2d(np.arange(image1[0].size), np.arange(image1[0].size), image1, kind='linear')
        interpoint = intpl.RectBivariateSpline(np.arange(image1.size/ image1[0].size),np.arange(image1[1].size), image1, kx = 5, ky=5)


        count = 0
        gi = -1
        gj = -1
        result = np.zeros((100,100))
        minDiff = np.inf
        minShift = [0,0]

        print "starting for set : ", set

        for i in np.arange(-0.5, .5, 0.01):
            gi+=1
            gj=-1
            for j in np.arange(-0.5, .5, 0.01):
                count += 1
                time1 = dt.datetime.now()
                gj+=1
                target = interpoint.__call__(np.arange((image1.size / image1[0].size))+i, np.arange(image1[0].size)+j)
                diff = target-image2
                diff = np.square(diff)
                diff = np.sum(diff)
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

        print "^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^"
        print "all done!"
        time2 = dt.datetime.now()
        print "time was : ", time2-time1
        print "Minimum point: ", np.min(result)
        #pt =  np.unravel_index(result.argmin(), (100,100))

        fname = "shift_resultsset1_%s.p" % (f1)
        pickle.dump(result, open(fname, "wb"))
        #print "at coords: ", pt
        print " at shift : " , minShift
        minImg = interpoint.__call__(np.arange((image1.size / image1[0].size))+minShift[0], np.arange(image1[0].size)+minShift[1])

        print "Maximum point: ", np.max(result)
        sub = (image2 - minImg)
        fname2 = "d_%s.p" % (f2)
        pickle.dump(sub, open(fname2, "wb"))
        isolate_cr.main(15,2001, sub)







if __name__ == '__main__':
    realInterp()






