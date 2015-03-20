'''
Created on Aug 22, 2012

@author: kahere
Modified from http://stackoverflow.com/questions/6652671/efficient-method-of-calculating-density-of-irregularly-spaced-points
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import time


# Boxsum smoothing function    
def boxsum(img, w, h, r):
    st = [0] * (w+1) * (h+1)
    for x in range(w):
        st[x+1] = st[x] + img[x]
    for y in range(h):
        st[(y+1)*(w+1)] = st[y*(w+1)] + img[y*w]
        for x in range(w):
            st[(y+1)*(w+1)+(x+1)] = st[(y+1)*(w+1)+x] + st[y*(w+1)+(x+1)] - st[y*(w+1)+x] + img[y*w+x]
    for y in range(h):
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        for x in range(w):
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)
            img[y*w+x] = st[y0*(w+1)+x0] + st[y1*(w+1)+x1] - st[y1*(w+1)+x0] - st[y0*(w+1)+x1]

def grid_density_boxsum(x0, y0, x1, y1, data):
    scale = 10 # scale factor allows finer resolution for smoothing filter
    w = (x1 - x0)*scale
    h = (y1 - y0)*scale
    r = 1 # must be an integer
    imgw = w
    imgh = h
    img = [0] * (imgw * imgh)
    for x, y in data:
        ix = int((x - x0) * scale)
        iy = int((y - y0) * scale)
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy * imgw + ix] += 1
    for p in range(4):
        boxsum(img, imgw, imgh, r)
    a = np.array(img).reshape(imgh,imgw)
    b = a[0:h,0:w]
    return b
    
def generate_graph():    
    # view area range
    view_ymin = 20
    view_ymax = 55
    view_xmin = -130
    view_xmax = -65
    # generate data
    xl = lon
    yl = lat
    
    # get visible data points
    xlvis = []
    ylvis = []
    for i in range(0,len(xl)):
        if view_xmin < xl[i] < view_xmax and view_ymin < yl[i] < view_ymax:
            xlvis.append(xl[i])
            ylvis.append(yl[i])

    
    # run boxsum smoothing, plot data
    t0 = time.clock()
    zd = grid_density_boxsum(view_xmin, view_ymin, view_xmax, view_ymax, zip(xl, yl))
    plt.title('boxsum smoothing - '+str(time.clock()-t0)+"sec")
    plt.imshow(zd, origin='lower', extent=[view_xmin, view_xmax, view_ymin, view_ymax])
#    plt.scatter(xlvis, ylvis) # show points in original dataset


if __name__=='__main__':
    
    # Import tornado data lat/lon from .csv file
    length = 57217
    data = np.empty([length,2])
    with open('1950_2011StartTracks.csv') as inf:
        data[:] = [(float(row['Lat']), row['Lon']) for row in csv.DictReader(inf)]
    lat = data[:,0]
    lon = data[:,1]
    
    # Plot results
    generate_graph()
    plt.show()
