#!/usr/bin/env python
# encoding: utf-8

'''
File: get_samples.py
Author: AFlock
Description: retrieve samples from a diff'd image, pickle them.

'''
def main():
    """
    MO:
    -Get the two matching images img1 and img2
    -subtract img1 - img2 = diff_img
    -consider only positive parts of diff_img
        -above a low threshold, these will be CR-positive pixels
        -take these slices from img1 NOT diff_img
    -get 1000 CR-positive examples, and 1000 non CR examples.
        -the non CR examples should include bright patches from img1
    -save these in a pickle



    """
    data_dir = "/misc/vlgscratch1/FergusGroup/abf277/hst"

if __name__ == '__main__':
    main()
