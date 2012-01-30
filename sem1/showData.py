#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.transforms as trsnfm
import numpy as np
import pyfits as pft
import pickle
import sys, os

def main():#{{{
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    filename_filter = 'raw'

    crval_dict = {}
    for data_file in os.listdir(data_dir):
        if filename_filter in data_file:
            print "V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V_V"

            print data_file

            raw_data = pft.open(data_dir + data_file)
            sci_data = raw_data[1].data

            print "Header:"
            crval1 = raw_data[1].header.get('CRVAL1')
            crval2 = raw_data[1].header.get('CRVAL2')
            print "CRVAL1: ", crval1
            print "CRVAL2: ", crval2
            crval_dict[data_file] = [crval1, crval2]

            raw_median = np.median(sci_data)

            shifted_array = np.roll(sci_data, 5, axis=0)
            shifted_array = np.roll(shifted_array, 5, axis=1)
            #print sci_data, shifted_array

            #sci_data = np.array([[5,1,1],[7,1,3,],[0,1,1]])
            #shifted_array = np.array([[2,2,8],[2,4,2,],[9,2,2]])
            difference_array = sci_data - shifted_array
            #print difference_array
            absolute_diff = np.absolute(difference_array)
            #print absolute_diff
            print "mean: " , np.mean(sci_data)
            print "median: ", np.median(sci_data)
            print "std: ", np.std(sci_data)
            shift_median = np.median(absolute_diff)
            print "median of shifted-original:", shift_median
            #imgplot = plt.imshow(sci_data, cmap = cm.gray, vmin=shift_median-1250 ,vmax=shift_median+1250)
            #imgplot = plt.imshow(sci_data, cmap = cm.gray)
            raw_data.close()

    for name, crlist1 in crval_dict.iteritems():
        for name2, crlist2 in crval_dict.iteritems():
            if name != name2:
                if (crlist1[1] - crlist2[1]) < .000002 and (crlist1[1] - crlist2[1]) < .00002:
                    print "%s is similar to %s: " % (name, name2)
                    print   "%s ::: %s" % (crval_dict[name2], crval_dict[name])

    pickle.dump(crval_dict, open("crvals.p", "wb"))
#}}}

def re_analyze():#{{{
    dict = pickle.load(open("crvals.p", "r"))
    for name, crlist1 in dict.iteritems():
        for name2, crlist2 in dict.iteritems():
            if name != name2:
                cr1_diff = abs(crlist1[0] - crlist2[0])
                cr2_diff = abs(crlist1[1] - crlist2[1])
                if cr1_diff < .000002 and cr2_diff < .000002:
                    print "%s is similar to %s: " % (name, name2)
                    print   "%s ::: %s" % (dict[name2], dict[name])
                    print "CRVAL1 diff: %s " % cr1_diff
                    print "CRVAL2 diff: %s " % cr2_diff
#}}}

def view_flat():
    pic = pft.open('/misc/vlgscratch1/FergusGroup/abf277/hst/j8m862gbq_flt.fits')
    data = pic[1].data
    zoomed_data = data[650:1200, 770:1250]

    median = np.median(data)
    imgplot = plt.imshow(zoomed_data, cmap = cm.gray, vmin =median-70, vmax =median+70)


def picture():#{{{
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    the_pictures = [['j8m862gbq_raw.fits','j8m81ccoq_raw.fits'],
            ['j8m862g8s_raw.fits','j8m81cckq_raw.fits'],
            ['j8m862gtq_raw.fits','j8m81cd8q_raw.fits'],
            [ 'j8m862goq_raw.fits','j8m81cd2q_raw.fits']]

    first_set = the_pictures[0]

    #for pic in first_set:
    pic = first_set[0]
    print "pic1: ", pic
    pic2 = first_set[1]
    print "pic2: ", pic2
    filename = data_dir + pic
    filename2 = data_dir + pic2
    raw_data = pft.open(filename)
    raw_data2 = pft.open(filename2)
    sci_data = raw_data[1].data
    sci_data2 = raw_data2[1].data

    print "Header 1:"
    crval1 = raw_data[1].header.get('CRVAL1')
    crval2 = raw_data[1].header.get('CRVAL2')
    print "CRVAL1: ", crval1
    print "CRVAL2: ", crval2

    print "Header 2:"
    crval1 = raw_data2[1].header.get('CRVAL1')
    crval2 = raw_data2[1].header.get('CRVAL2')
    print "CRVAL1: ", crval1
    print "CRVAL2: ", crval2

    raw_median = np.median(sci_data)

    shifted_array = np.roll(sci_data, 5, axis=0)
    shifted_array = np.roll(shifted_array, 5, axis=1)
    #print sci_data, shifted_array

    #calculate MAD
    dif = sci_data - shifted_array
    absdif = np.absolute(dif)
    mad = np.median(absdif)

    print "%s: " % pic
    print "mean: " , np.mean(sci_data)
    print "median: ", np.median(sci_data)
    print "std: ", np.std(sci_data)
    print "min: ", np.min(sci_data)
    print "max: ", np.max(sci_data)

    print "%s: " % pic2
    print "mean: " , np.mean(sci_data2)
    print "median: ", np.median(sci_data2)
    print "std: ", np.std(sci_data2)
    print "min: ", np.min(sci_data2)
    print "max: ", np.max(sci_data2)

    difference_array = sci_data - sci_data2
    absolute_diff = np.absolute(difference_array)
    print "absolute difference: "
    print "mean: " , np.mean(absolute_diff)
    print "median: ", np.median(absolute_diff)
    print "std: ", np.std(absolute_diff)
    print "min: ", np.min(absolute_diff)
    print "max: ", np.max(absolute_diff)
    #shift_median = np.median(absolute_diff)
    #print "MAD:", mad



    print "difference_array: "
    print "mean: " , np.mean(difference_array)
    print "median: ", np.median(difference_array)
    print "std: ", np.std(difference_array)
    print "min: ", np.min(difference_array)
    print "max: ", np.max(difference_array)
    #imgplot = plt.imshow(sci_data, cmap = cm.gray, vmin=np.median(sci_data)-10*shift_median ,vmax=np.median(sci_data)+10*shift_median)
    raw_data.close()
    raw_data2.close()

    #print sci_data
    #print "size: ", sci_data.size
    #zoomed_img = difference_array[650:1200, 770:1250]
    #imgplot = plt.imshow(zoomed_img, cmap = cm.gray, vmin=-5*mad ,vmax=5*mad)

#}}}

def histogram():
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    the_pictures = ['j8m862gbq_raw.fits','j8m81ccoq_raw.fits']


    for pic in the_pictures:
        print pic
        filename = data_dir + pic
        raw_data = pft.open(filename)
        sci_data = raw_data[0]
        header = sci_data.header
        print header
        """
        for key, value in header.iteritems():
            print "%s : %s" % (key, value)
        """

    """
    sci_data_flat = sci_data.ravel()
    sci_data_flat = np.log(sci_data_flat)
    #sci_data_flat = np.log(sci_data_flat)
    #sci_data_flat = np.log(sci_data_flat)
    plt.hist(sci_data_flat, bins=150)
    plt.show()
    """

if __name__ == '__main__':
    #main()
    #re_analyze()
    picture()
    #histogram()

    #view_flat()
