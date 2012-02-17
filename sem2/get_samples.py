#!/usr/bin/env python
# encoding: utf-8

'''
File: get_samples.py
Author: AFlock
Description: retrieve samples from a diff'd image, pickle them.
'''

import os, pickle
import pyfits            as pft
import numpy             as np
import matplotlib.cm     as cm
import matplotlib.pyplot as plt
import scipy.interpolate as intpl
from optparse                import OptionParser
from mpl_toolkits.axes_grid1 import ImageGrid

data_dir = "/misc/vlgscratch1/FergusGroup/abf277/hst"
xshift = 0.1
yshift = -0.01

def errorWise(shifted, comparison, mad, epsilon):
    diff = shifted-comparison
    diff = np.square(diff)
    diff = np.sum(diff)
    epmad = (epsilon* np.square(mad))
    diff = diff/(diff+ epmad)
    return diff

def main(options, args):
    """#{{{
    MO:
    -Get the two matching images img1 and img2
    -subtract img1 - img2 = diff_img
    -consider only positive parts of diff_img
        -above a low threshold, these will be CR-positive pixels
        -take these slices from img1 NOT diff_img
    -get 1000 CR-positive examples, and 1000 non CR examples.
        -the non CR examples should include bright patches from img1
    -save these in a pickle

    Data format for samples:
    {x:[flattened array] y:1/0}
    """#}}}
    #get pixel width
    print options
    p = options.pixel

    original_file_path = '/j8m81cd8q_raw.fits'
    original_file_data = pft.open(data_dir+original_file_path)
    original_file_data = original_file_data[1].data

    match_file_path = '/j8m862gtq_raw.fits'
    match_file_data = pft.open(data_dir+match_file_path)
    match_file_data = match_file_data[1].data

    #crop the image for ease of math
    s = original_file_data.size/original_file_data[0].size
    original_file_data =  original_file_data[0:s, 0:s]
    match_file_data =  match_file_data[0:s, 0:s]

    #calculate mad for use in error estimates
    diff = original_file_data - match_file_data
    absdif = np.absolute(diff)
    mad = np.median(absdif)

    #interpolate across the match image so we can shift it
    #if we haven't pre-calculated this, need to calculate best appropriate shife
    for file in os.listdir('.'):
        if "shifted_image" in file:
            print "pre-shifted file detected", file
            break
    else:
        #need to calculate it and save
        #iterate through pixel shifts by .01 steps and seem how much 'error' is implied
        #lowest error wins
        print "calculating shifted image"
        min_diff = np.inf
        min_shift = [0,0]
        interpolation = intpl.RectBivariateSpline(np.arange(s),
                np.arange(s), match_file_data, kx = 5, ky=5)
        for i in np.arange(-0.5, .5, 0.01):
            for j in np.arange(-0.5, .5, 0.01):
                target = interpolation.__call__(np.arange(s)+i, np.arange(s)+j)
                diff = errorWise(target, original_file_data, mad, 1)
                if diff < min_diff:
                    print "%s :: %s" % (i, j)
                    min_diff = diff
                    min_shift[0] = i
                    min_shift[1] = j
        shifted_image = interpolation.__call__(np.arange(s)+min_shift[0], np.arange(s)+min_shift[1])
        #pickle the shifted image so we don't need to calculate it again
        print "best shift was : ", min_shift
        pickle.dump(shifted_image, open("shifted_image_%s.p" % (match_file_path[1:10]), "wb"))

    for file in os.listdir('.'):
        if "shifted_image" in file:
            shifted_image = pickle.load(open(file, "rb"))
            break
    else:
        print "hmm no shifted image something's wrong"

    #subtract the images
    subtracted_image = original_file_data - shifted_image

    #hacky way to see distortion stats#{{{
    """
    dist_sec = subtracted_image[1000:1060, 1010:1070]

    print "med: ", np.median(dist_sec)
    m =  np.median(dist_sec)
    print "max: ", np.max(dist_sec)
    print "min: ", np.min(dist_sec)
    plt.imshow(dist_sec, cmap=cm.gray, vmin = m-200, vmax = m+200)
    plt.colorbar()
    return

    """
    #end hax#}}}

    print "Subtracted image stats"
    print "med: ", np.median(subtracted_image)
    print "max: ", np.max(subtracted_image)
    print "min: ", np.min(subtracted_image)
    sub_absdif = np.absolute(subtracted_image)
    sub_mad = np.median(sub_absdif)
    orig_absdif = np.absolute(original_file_data)
    orig_mad = np.median(orig_absdif)
    print "mad: ", sub_mad

    """
    #I'd like to see the image
    plt.imshow(subtracted_image, cmap=cm.gray, vmin=0, vmax=sub_mad+200)
    return
    """

    #To keep track of CR-positive samples
    sample_count = 0
    cr_coords = []
    positive_samples = []

    #Find Cosmic Rays!
    for row_num, row in enumerate(subtracted_image):
        for p_num, pixel in enumerate(row):
            in_bounds = p_num > p/2 and row_num > p/2 and row_num < s-(p/2) and p_num < s-(p/2)
            if pixel > 200 and in_bounds:
                slice_coords = ((row_num-p/2, row_num+p/2+2), (p_num-p/2, p_num+p/2+2))
                #print "CR found %s", pixel
                """
                fig = plt.figure(1, (1., 3.))
                grid = ImageGrid(fig, 111, nrows_ncols = (1, 3), axes_pad=0.1)

                im_s = match_file_data[slice_coords[0][0]:slice_coords[0][1],
                        slice_coords[1][0]:slice_coords[1][1]]
                im_u = subtracted_image[slice_coords[0][0]:slice_coords[0][1],
                        slice_coords[1][0]:slice_coords[1][1]]
                """

                if sample_count < 1000: #only want to save 1000
                    print slice_coords, sample_count
                    im_o = original_file_data[slice_coords[0][0]:slice_coords[0][1],
                            slice_coords[1][0]:slice_coords[1][1]]
                    """#{{{
                    grid[0].imshow(im_u, cmap=cm.gray)#, vmin=0, vmax=sub_mad+70)
                    grid[1].imshow(im_o, cmap=cm.gray)#, vmin=orig_mad-70, vmax=orig_mad+70)
                    grid[2].imshow(im_s, cmap=cm.gray)#, vmin=orig_mad-70, vmax=orig_mad+70)
                    #plt.savefig('%s/samples/%s_%s.png' % (data_dir, row_num, p_num))
                    """#}}}

                    #put the slice from the original into a nice format
                    formatted_sample = {'x': im_o.flatten(), 'y': 1}
                    #print formatted_sample
                    positive_samples.append(formatted_sample)
                    sample_count += 1

                if sample_count == 1000:
                    break
                # still want to mark those coordinates as taken
                cr_coords.append((row_num, p_num))

    #Now we want 1000 CR-negative samples -> get the highest pixels that were NOT marked as CRs
    pixel_list = []
    negative_samples = []
    for row_num, row in enumerate(original_file_data):
        for p_num, pixel in enumerate(row):
            if (row_num, p_num) not in cr_coords:
                print "add to poss neg", row_num, p_num
                pixel_list.append([(row_num, p_num), pixel])
    print "pre sort"
    top_1000 = sorted(pixel_list, key=lambda pair: pair[1])[-1000:]

    print "Negative sample gathering"
    for s in top_1000:
        row_num, p_num = s[0]
        #saving the sample as data
        sl_co = ((row_num-p/2, row_num+p/2+2), (p_num-p/2, p_num+p/2+2))
        print sl_co
        slice = original_file_data[sl_co[0][0]:sl_co[0][1], sl_co[1][0]:sl_co[1][1]]
        formatted_slice = {'x': slice.flatten(), 'y':0}
        negative_samples.append(formatted_slice)

        #saving the three images sliced as per the coordinates of the sample
        """#{{{
        im_o = original_file_data[sl_co[0][0]:sl_co[0][1], sl_co[1][0]:sl_co[1][1]]
        im_s = match_file_data[sl_co[0][0]:sl_co[0][1], sl_co[1][0]:sl_co[1][1]]
        im_u = subtracted_image[sl_co[0][0]:sl_co[0][1], sl_co[1][0]:sl_co[1][1]]

        fig = plt.figure(1, (1., 3.))
        grid = ImageGrid(fig, 111, nrows_ncols = (1, 3), axes_pad=0.1)
        grid[0].imshow(im_u, cmap=cm.gray)#, vmin=0, vmax=sub_mad+70)
        grid[1].imshow(im_o, cmap=cm.gray)#, vmin=orig_mad-70, vmax=orig_mad+70)
        grid[2].imshow(im_s, cmap=cm.gray)#, vmin=orig_mad-70, vmax=orig_mad+70)
        #plt.savefig('%s/samples/%s_%s.png' % (data_dir, row_num, p_num))
        """#}}}

    #now dump all the samples together
    print negative_samples[0:3]
    print positive_samples[0:3]

    print "pickling %s negative samples" % len(negative_samples)
    print "pickling %s positive samples" % len(positive_samples)

    pickle.dump(negative_samples, open("%s/samples/negative.p" % data_dir, "wb"))
    pickle.dump(positive_samples, open("%s/samples/positive.p" % data_dir, "wb"))



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--pixel", dest="pixel",
            help ="how wide in pixels should the samples be", default=9)
    (options, args) = parser.parse_args()
    main(options, args)
