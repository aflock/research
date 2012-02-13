'''
File: isolate_cr.py
Author: AF
Description: Generate NxN  slices of a diffed image,
            and set everything below a threshold to 0
'''
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np
import pickle
import sys, os
import pyfits as pft

def main(n , thresh, sub_thresh, orig=None):
    print "start isolate cr"
    data_dir = "/misc/vlgscratch1/FergusGroup/abf277/hst"
    '''
    filename = "pickles/diffedpic_j8m862g8s_raw.fits.p"
    if orig == None:
        l = pickle.load(open(filename, "rb"))
    else:
        l = orig
        #want orig's original
        origfile = '/j8m81cd8q_raw.fits'
        raw = pft.open(datadir+origfile)
        data = raw[1].data
    '''

    origfile = '/j8m81cd8q_raw.fits'
    raw = pft.open(data_dir+origfile)
    data = raw[1].data


    sub_data = data[800:1200, 800:1200]
    l = pickle.load(open("d_j8m81cd8q_raw.fits.p", "rb"))
    o = np.copy(l)
    sub_o = o[800:1200, 800:1200]
    off = n/2 + 1
    sub_o = sub_o[off:-off, off:-off]
    o = o[off:-off, off:-off]

    absdif = np.absolute(data)
    mad = np.median(absdif)

    #let's set everything below 0 to 0
    for i, row in enumerate(l):
        for j, el in enumerate(row):
            if el < 0:
                l[i,j] = 0


    #normalize and set mad
    """
    max = np.median(sub_data)
    sub_data = sub_data*100/max
    sub_o = sub_o*100/max
    """
    absdif = np.absolute(sub_data)
    mad = np.median(absdif)

    """
    for i, row in enumerate(sub_data):
        for j, el in enumerate(row):
            if el > 3000:
                sub_data[i,j]= 3000
    plt.hist(sub_data, bins = 200,  log=True)
    for i, row in enumerate(sub_o):
        for j, el in enumerate(row):
            if el > 10000:
                sub_o[i,j]= 10000
    """



    #plt.hist(sub_o, bins = 200,  range = (2000, 10000), log=True)
    #plt.hist(sub_o, bins = 200,  log=True)
    #return

    print "max: ", np.max(sub_data)
    print "min: ", np.min(sub_data)
    print "med: ", np.median(sub_data)

    #plt.imshow(sub_data, cmap=cm.gray, vmin=mad-70, vmax=mad+70)
    return
    #plt.imshow(l, cmap=cm.gray, vmin=mad-70, vmax=mad+70)

    #a smaller image to test on
    slices = []
    uslices = []
    count  = 0 #keep track of CR likely
    count2 = 0 #keep track of under thresh
    ofs = n/2
    for i, row in enumerate(sub_o):
        for j, el in enumerate(row):
            if el > thresh:
                if count < 125:
                    #create an NxN slice from it
                    slice = sub_data[i+off-ofs:i+off+ofs, j+off-ofs:j+off+ofs]
                    #plt.imshow(slice, cmap=cm.gray )
                    #plt.imshow(slice, cmap=cm.gray, vmin=mad-70, vmax=mad+70)
                    #return
                    #plt.imshow(slice, cmap=cm.gray, vmin=mad-70, vmax=mad+70)
                    #break
                    slices.append(slice)
                    """
                    print el
                    print slice
                    plt.imshow(slice, cmap=cm.gray)
                    return
                    """
                    plt.imsave('%s/cr_%s%s.png' % (data_dir, i, j), slice, cmap=cm.gray, vmin=mad-70, vmax=mad+70)
                    """
                    slice = (slice/slice.max())*255
                    f = open('%s/slice%s%s.png' % (data_dir, i, j), 'wb', )
                    w = png.Writer(255, 1, greyscale=True)
                    w.write(f, slice)
                    f.close()
                    """
                    print "found %s CR slices" % count
                    count +=1
            elif el > sub_thresh:
                if count2 < 125:
                    #create an NxN slice from it
                    slice = sub_data[i+off-ofs:i+off+ofs, j+off-ofs:j+off+ofs]
                    #plt.imshow(slice, cmap=cm.gray, vmin=mad-70, vmax=mad+70)
                    #break
                    uslices.append(slice)
                    plt.imsave('%s/sub_%s%s.png' % (data_dir, i, j), slice, cmap=cm.gray, vmin=mad-70, vmax=mad+70)
                    print "found %s likely sub slices" % count2
                    count2 +=1



    #save all slices
    pickle.dump(slices, open("%s/crSlices_%s_%s.p" % (data_dir, thresh, sub_thresh), "wb"))
    pickle.dump(uslices, open("%s/uSlices_%s_%s.p" % (data_dir, thresh, sub_thresh), "wb"))


def quick_test():


    data_dir = "/misc/vlgscratch1/FergusGroup/abf277/hst"
    origfile = '/j8m81cd8q_raw.fits'
    raw = pft.open(data_dir+origfile)
    data = raw[1].data
    sub_data = data[210:250, 220:250]

    med = np.median(sub_data)
    print med
    print " max :", np.max(sub_data)
    print " min :", np.min(sub_data)
    print sub_data

    print "normalizing"
    sub_data = sub_data*100/np.max(sub_data)
    med = np.median(sub_data)
    print med
    print " max :", np.max(sub_data)
    print " min :", np.min(sub_data)
    print sub_data

    """
    sub_data = data[110:150, 0:50]
    med = np.median(sub_data)
    print med
    print " max :", np.max(sub_data)
    print " min :", np.min(sub_data)
    print sub_data
    """

    print "second sub"
    sub_data = data[110:150, 0:50]

    med = np.median(sub_data)
    print med
    print " max :", np.max(sub_data)
    print " min :", np.min(sub_data)
    print sub_data

    print "normalizing"
    sub_data = sub_data*100/np.max(sub_data)
    med = np.median(sub_data)
    print med
    print " max :", np.max(sub_data)
    print " min :", np.min(sub_data)
    print sub_data
if __name__ == '__main__':
    """
    quick_test()

    """
    if len(sys.argv) < 2:
        #size of slice, 1st threshold for CR, second threshold for edge cases, original file
        main(19, 105.5, 101.1, "")
    else:
        main(int(sys.argv[1]), int(sys.argv[2]))
