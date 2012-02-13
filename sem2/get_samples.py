#!/usr/bin/env python
# encoding: utf-8

'''
File: get_samples.py
Author: AFlock
Description: retrieve samples from a diff'd image, pickle them.
'''

import pickle
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

    #interpolate across the match image so we can shift it
    interpolation = intpl.RectBivariateSpline(np.arange(s),
            np.arange(s), match_file_data, kx = 5, ky=5)
    shifted_image = interpolation.__call__(np.arange(s)+xshift, np.arange(s)+yshift)

    #subtract the images
    subtracted_image = original_file_data - shifted_image

    print "Subtracted image stats"
    print "med: ", np.median(subtracted_image)
    print "max: ", np.max(subtracted_image)
    print "min: ", np.min(subtracted_image)
    sub_absdif = np.absolute(subtracted_image)
    sub_mad = np.median(sub_absdif)
    orig_absdif = np.absolute(subtracted_image)
    orig_mad = np.median(orig_absdif)
    print "mad: ", sub_mad

    #Find Cosmic Rays!
    for row_num, row in enumerate(subtracted_image):
        for p_num, pixel in enumerate(row):
            if pixel > 100 and p_num > p/2 and row_num > p/2:
                slice_coords = ((row_num-p/2, row_num+p/2+2), (p_num-p/2, p_num+p/2+2))
                print slice_coords
                #print "CR found %s", pixel
                fig = plt.figure(1, (1., 3.))
                grid = ImageGrid(fig, 111, nrows_ncols = (1, 3), axes_pad=0.1)

                im_o = original_file_data[slice_coords[0][0]:slice_coords[0][1], slice_coords[1][0]:slice_coords[1][1]]
                im_s = shifted_image[slice_coords[0][0]:slice_coords[0][1], slice_coords[1][0]:slice_coords[1][1]]
                im_u = subtracted_image[slice_coords[0][0]:slice_coords[0][1]  , slice_coords[1][0]:slice_coords[1][1]]
                grid[0].imshow(im_u, cmap=cm.gray, vmin=0, vmax=sub_mad+70)
                grid[1].imshow(im_o, cmap=cm.gray, vmin=orig_mad-70, vmax=orig_mad+70)
                grid[2].imshow(im_s, cmap=cm.gray, vmin=orig_mad-70, vmax=orig_mad+70)
                plt.show(cm)
                return

    """
    plt.imshow(subtracted_image, cmap=cm.gray, vmin=0, vmax=mad+700)
    plt.imshow(original_file_data, cmap=cm.gray, vmin=0, vmax=mad+700)
    plt.imshow(match_file_data, cmap=cm.gray, vmin=0, vmax=mad+700)
    for row in subtracted_image:
        for pixel in row:
            if pixel > 50:
                #print "CR found %s", pixel
                pass
    """


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--pixel", dest="pixel",
            help ="how wide in pixels should the samples be", default=9)
    (options, args) = parser.parse_args()
    main(options, args)
