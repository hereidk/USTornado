'''
Created on Aug 21, 2012

@author: Kelly Hereid
'''

import numpy as np
import random
import math
import bisect

class Tornado(object):
    '''
    classdocs
    '''
    
    SCALE = 10
    
    
    
    def __init__(self):
        '''
        Constructor
        '''

        self.intensity = self.intensity()
        self.intensity = int(self.intensity[0])
        self.bearing = self.bearing()
        self.startpt = self.startpt(-130, 20, -65, 55)
        self.pathlength = self.pathlength(self.startpt)
        self.endpt = self.endpt()
    
    #Set intensity of tornado (F3, F4, F5)
    def intensity(self):
        scale = self.SCALE
        intensity_dist = np.empty([100 * scale,1]);
        f3_freq = 78 * scale;
        f4_freq = 20 * scale;
        f5_freq = 2 * scale;
        intensity_dist[:f3_freq] = 3;
        intensity_dist[f3_freq+1:f3_freq+f4_freq] = 4;
        intensity_dist[f3_freq+f4_freq+1:f3_freq+f4_freq+f5_freq] = 5;
        intensity = random.sample(list(intensity_dist),1);
        return intensity[0]
        
    #Set length of tornado path based on intensity
    def pathlength(self, startpt):
        startlat = startpt[0]
#        startlon = startpt[1]
        length_lon = (math.cos(math.radians(startlat))) * 69.2
        f3_path_mean = 18.0418 / length_lon
        f3_path_std = 17.7501 / length_lon
        f4_path_mean = 29.1772 / length_lon
        f4_path_std = 29.2617 / length_lon
        f5_path_mean = 38.4829 / length_lon
        f5_path_std = 34.2547 / length_lon
        
        pathlength = 0
        
        if self.intensity == 3:
            pathlength = random.gauss(f3_path_mean,f3_path_std)
        elif self.intensity == 4:
            pathlength = random.gauss(f4_path_mean,f4_path_std)
        elif self.intensity == 5:
            pathlength = random.gauss(f5_path_mean,f5_path_std)
            
        if pathlength < 0:
            pathlength = 0.001;
        return pathlength
        
    # Set bearing of tornado based on historical distribution
    def bearing(self):
        bearing = np.random.exponential(23.72520339382)
#        bearing = 90 - bearing;
        return bearing
        
    def startptdist(self):
        pass
        
#    Define starting point of tornado based on weighted random probability distribution from historical tornado density
    def startpt(self, x0, y0, x1, y1):
#        Weighted random function after http://docs.python.org/py3k/library/random.html
        zd = self.generate_grid()
        cumdist = list(accumulate(zd.flat))
        x = random.random() * cumdist[-1]
        loc = range(0,len(zd.flat))
        startpt = loc[bisect.bisect(cumdist, x)]
        scale = float(self.SCALE)
        w = float((x1 - x0) * scale)
        startlat = (startpt/w)/scale + float(y0)
        startlon = startpt%w/scale + float(x0)
        startpt = [startlat, startlon]
        return startpt
        
    #Calculate endpoint of tornado path from startpoint, length, bearing
    def endpt(self):
        path = self.pathlength
        endlat = path * math.sin(math.radians(self.bearing));
        endlon = path * math.cos(math.radians(self.bearing));
        endlat = endlat + self.startpt[0];
        endlon = endlon + self.startpt[1];
        endpt = [endlat, endlon]
        return endpt
      
    
    #Import tornado data lat/lon from .csv file  
    def import_data(self):
        data = np.loadtxt('1950_2011StartTracks.csv',delimiter=',')
        lat = data[:,0]
        lon = data[:,1]
        return lat, lon
   
    # Boxsum smoothing function    
    def boxsum(self, img, w, h, r):
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

    def grid_density_boxsum(self, x0, y0, x1, y1, data):
        scale = 10 # scale factor allows finer resolution for smoothing filter
        w = (x1 - x0)*scale
        h = (y1 - y0)*scale
        r = 2 # must be an integer
        imgw = w
        imgh = h
        img = [0] * (imgw * imgh)
        for x, y in data:
            ix = int((x - x0) * scale)
            iy = int((y - y0) * scale)
            if 0 <= ix < imgw and 0 <= iy < imgh:
                img[iy * imgw + ix] += 1
        for p in range(4):
            self.boxsum(img, imgw, imgh, r)
        a = np.array(img).reshape(imgh,imgw)
        b = a[0:h,0:w]
        return b
        
    #Generate grid of tornado strike likelihood based on density of historical tornado startpoints
    def generate_grid(self):    
        # view area range
        view_ymin = 20
        view_ymax = 55
        view_xmin = -130
        view_xmax = -65
        # generate data
        [lat, lon] = self.import_data()
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
#        t0 = time.clock()
        zd = self.grid_density_boxsum(view_xmin, view_ymin, view_xmax, view_ymax, zip(xl, yl))
#        plt.title('boxsum smoothing - '+str(time.clock()-t0)+"sec")
#        plt.imshow(zd, origin='lower', extent=[view_xmin, view_xmax, view_ymin, view_ymax])
#        plt.scatter(xlvis, ylvis) # show points in original dataset
        return zd

def accumulate(iterable):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    it = iter(iterable)
    total = int(next(it)) # Change from long to int
    yield total
    for element in it:
        total = total + int(element) # Change from long to int
        yield total

#    plt.show()
