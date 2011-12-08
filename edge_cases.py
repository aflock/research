'''
File: edge_cases.py
Author: AF
Description: Generate samples of "high but not CR" pixels
    -Use the Diffed image for CR identification (taking 95% of signal to be CR)
    -Remaining pixels are indexed by intesnity, top 200 are sliced.
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np
import pickle
import operator
import sys, os
import pyfits as pft


def main(n):
    off = n/2
    data_dir = "/misc/vlgscratch1/FergusGroup/abf277/hst"
    origfile = '/j8m81cd8q_raw.fits'
    raw = pft.open(data_dir+origfile)

    raw_data = raw[1].data
    subd_img = pickle.load(open("d_j8m81cd8q_raw.fits.p", "rb"))
    mark_cr = np.zeros(np.size(raw_data))


    if True:    #if you want to cut the image down for testing
        raw_data = raw_data[800:1200, 800:1200]
        subd_img = subd_img[800:1200, 800:1200]
        mark_cr = np.zeros((400,400))


    s = subd_img[200:250 , 212:250]
    print "max: ", np.max(s)
    print "min: ", np.min(s)
    print "med: ", np.median(s)


    """
    m = np.median(s)
    #plt.imshow(s, cmap = cm.gray, vmin=m-70, vmax=m+70 )
    plt.imshow(subd_img, cmap = cm.gray, vmin = 400, vmax = 800)
    plt.colorbar()
    return
    """


    med = np.median(raw_data)
    print "raw med: ", med
    #normalizing the sub image is a bad idea
    #subd_img = subd_img*100/med

    print "max: ", np.max(subd_img)
    print "min: ", np.min(subd_img)
    print "med: ", np.median(subd_img)


    list_of_free = {}
    for i, row in enumerate(subd_img):
        for j, el in enumerate(row):
            if el > 600:
                mark_cr[i,j] = 1
            else:
                list_of_free[(i,j)] = raw_data[i,j]



    sorted_list = sorted(list_of_free.iteritems(), key=operator.itemgetter(1))
    sorted_list.reverse()
    count = 0
    while count < 100:
        coords = sorted_list[count][0]
        i = coords[0]
        j = coords[1]
        slice = raw_data[i-off:i+off+1, j-off:j+off+1]
        print np.size(slice)
        if np.size(slice) == n*n:
            print "found %s suspects" % count
            med = np.median(slice)
            plt.imsave('%s/sub_%s%s.png' % (data_dir, i, j), slice, cmap=cm.gray, vmin=med-70, vmax=med+70)
        count += 1

if __name__ == '__main__':
    main(15)
