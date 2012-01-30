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

def testInt():#{{{
    a = np.zeros((9,9))
    a[4,4] = 1
    interpoint = intpl.RectBivariateSpline(np.array(range(9)),np.array(range(9)),a, kx =1, ky =1)
    target = np.zeros((101,101))
    lin = intpl.interp2d(np.array(range(9)), np.array(range(9)), a, kind='linear')
    cub = intpl.interp2d(np.array(range(9)), np.array(range(9)), a, kind='cubic')
    print "RectBivariateSpline approx of coord 4,4: ", interpoint.__call__(4,4)
    print "cubic approx of coord 4,4: ", cub.__call__(4,4)
    print "linear approx of coord 4,4: ", lin.__call__(4,4)
    print "XXXX(X9X(x((X(X(X(X(X(X((X(X(X("

    print "RectBivariateSpline approx of coord 4,4.05: ", interpoint.__call__(4,4.05)
    print "cubic approx of coord 4,4.05: ", cub.__call__(4,4.05)
    print "linear approx of coord 4,4.05: ", lin.__call__(4,4.05)

    print "XXXX(X9X(x((X(X(X(X(X(X((X(X(X("

    print "RectBivariateSpline approx of coord 4,4.10: ", interpoint.__call__(4,4.10)
    print "cubic approx of coord 4,4.10: ", cub.__call__(4,4.10)
    print "linear approx of coord 4,4.10: ", lin.__call__(4,4.10)

    print "XXXX(X9X(x((X(X(X(X(X(X((X(X(X("

    print "RectBivariateSpline approx of coord 4,4.2: ", interpoint.__call__(4,4.2)
    print "cubic approx of coord 4,4.2: ", cub.__call__(4,4.2)
    print "linear approx of coord 4,4.2: ", lin.__call__(4,4.2)
    """
    gi = 0
    gj = 1

    for i in np.arange(-0.5, 0.5, 0.01):
        for j in np.arange(-0.5, 0.5, 0.01):
            a = interpoint.__call__(4+i,4+j)
            c = np.abs(a - 1)
            target[gj,gi] = c
            gj += 1
        gj = 0
        gi += 1
    """

    imgplot = plt.imshow(target, cmap=cm.hot)
    plt.colorbar()
#}}}

def reAnalyze():
    result = pickle.load( open("shift_results.p", "rb"))
    plt.imshow(result, cmap = cm.hot)
    plt.colorbar();
    print np.min(result)
    pt =  np.unravel_index(result.argmin(), (150,50))
    print pt

def realInterp():
    files = ['j8m862gbq_raw.fits','j8m81ccoq_raw.fits']
    f1 = files[0]
    f2 = files[1]
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    filename = data_dir + f1
    filename2 = data_dir + f2
    raw_data = pft.open(filename)
    raw_data2 = pft.open(filename2)
    sci_data = raw_data[1].data
    sci_data2 = raw_data2[1].data
    #choose a 100x100 grid to interpolate on
    pt1 = 0
    pt2 = 2000
    image1 = sci_data[pt1:pt2, pt1:pt2]
    image2 = sci_data2[pt1:pt2, pt1:pt2]

    #image1 = sci_data
    #image2 = sci_data2

    #image1 = np.zeros((5,5))
    #image2 = np.zeros((5,5))
    #image1[2,2] = 1
    #image2[1,1] = 1
    diff = image1 - image2
    median = np.median(diff)
    #imgplot = plt.imshow(diff, cmap = cm.gray, vmin =median-70, vmax =median+70)
    #plt.imshow(image1, cmap = cm.gray)
    print image1[0].size
    print image1[1].size
    #interpolate across image 1
    print "lets interpolate!"
    time1 = dt.datetime.now()

    #interpoint = intpl.interp2d(np.arange(image1[0].size), np.arange(image1[0].size), image1, kind='linear')
    interpoint = intpl.RectBivariateSpline(np.arange(image1[0].size),np.arange(image1[1].size), image1, kx = 5, ky=5)
    """
    minImg = interpoint.__call__(np.arange(image1[0].size)+.1, np.arange(image1[0].size))

    sub = (minImg-image2)
    plt.imshow(minImg-image2, cmap = cm.gray , vmin =median-70, vmax =median+70)

    flatten = sub.ravel()

    flatten = np.log(flatten)
    #pl"t.hist(flatten, bins = 200)

    """

    print "all done!"
    time2 = dt.datetime.now()
    print "time was : ", time2-time1

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
            diff = error(target, image2)
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
    print "Minimum point: ", np.min(result)
    #pt =  np.unravel_index(result.argmin(), (100,100))

    pickle.dump(result, open("shift_resultsset1.p", "wb"))
    #print "at coords: ", pt
    print "shift : " , minShift
    minImg = interpoint.__call__(np.arange(image1[0].size)+minShift[0], np.arange(image1[0].size)+minShift[1])
    print "Maximum point: ", np.max(result)
    sub = (minImg - image2)
    pickle.dump(sub, open("diffedpic1.p", "wb"))
    #plt.imshow(result, cmap = cm.hot)
    median = np.median(minImg- image2)
    plt.imshow(sub, cmap = cm.gray , vmin =median-70, vmax =median+70)
    plt.colorbar()

    """
    for k in np.arange(image1[0].size):
        for l in np.arange(image1[0].size):
            #target[k,l] = interpoint.__call__(k,l)
            target[k,l] = image1[k,l]
    """


    """
    gi = -1
    gj = -1
    for i in np.arange(-0.5, 0.5, 0.1):
        gi +=1
        gj = -1
        for j in np.arange(-0.5, 0.5, 0.1):
            gj +=1
            time3 = dt.datetime.now()
            count += 1
            for k in np.arange(image1[0].size):
                for l in np.arange(image1[0].size):
                    target[k,l] = interpoint.__call__(k+i,l+j)
            diff = target - image2
            diff = np.square(diff)
            diff = np.sum(diff)
            result[gi,gj] = diff
            time2 = dt.datetime.now()
            print "time for interation ", count, " : ", time2-time3
            print "diff is ", diff, "for x change: ", i, " :: ychange : ", j
            print "^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^"
            print "^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^_V_^"




    print "all done!"
    time2 = dt.datetime.now()
    print "total time was : ", time2-time1
    imgplot = plt.imshow(result, cmap=cm.hot)
    plt.colorbar()
    """


def main():
    files = ['j8m862gbq_raw.fits','j8m81ccoq_raw.fits']
    f1 = files[0]
    f2 = files[1]
    data_dir = '/misc/vlgscratch1/FergusGroup/abf277/hst/'
    filename = data_dir + f1
    filename2 = data_dir + f2
    raw_data = pft.open(filename)
    raw_data2 = pft.open(filename2)
    sci_data = raw_data[1].data
    sci_data2 = raw_data2[1].data
    pts = []
    #choose 1000 random points to interpolate on
    for i in range(1,1000):
        num1 = np.random.randint(0, 1997)
        num2 = np.random.randint(0, 3997)
        point = [num1, num2]
        pts.append(point)

    #construct subplots with values from f1's image in the right coords
    subPlots = []
    pointCount = 0
    for point in pts:
        pointCount+=1
        blank = np.zeros((7,7))
        #fill array with values from f1
        #add to subplots, with a tag for orig image? (point and subplot will order the same, no need)
        for i in range(7):
            for j in range(7):
                blank[i,j] = sci_data[point[0]+i, point[1]+j]


        target = np.zeros((100,100))
        print pointCount
        #print target
        interpoint = intpl.interp2d(np.array(range(7)), np.array(range(7)), blank, kind='linear')
        #print interpoint
        x, y = np.mgrid[-0.5:.01:0.5j,-0.5:.01:0.5j]
        #znew = intpl.RectBivariateSpline(np.array(range(5)),np.array(range(5)),blank, kx =1, ky =1)
        gi= 0
        gj= 0
        for i in np.arange(-0.5, 0.5, 0.01):
            for j in np.arange(-0.5, 0.5, 0.01):
                a = interpoint.__call__(3+i,3+j)
                c = np.abs(a - sci_data2[point[0]+3, point[1]+3])
                target[gi,gj] = c
                gj += 1
            gj=0
            gi += 1

        print target
        subPlots.append(target)

    average = np.zeros((100,100))
    count = 0
    for plot in subPlots:
        count+=1
        average += plot
    average = average/count
    imgplot = plt.imshow(average, cmap=cm.hot)

    #znew = intpl.RectBivariateSpline(np.array(range(3)),np.array(range(3)),blank)
    #plt.figure()
    #plt.pcolor(x,y,znew)
    plt.colorbar()
    #plt.show()




if __name__ == '__main__':
    #realInterp()
    reAnalyze()






