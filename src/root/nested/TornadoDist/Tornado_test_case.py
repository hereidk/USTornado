'''
Created on Aug 21, 2012

@author: kahere
'''
from root.nested.Tornado import Tornado
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':
    view_ymin = 20
    view_ymax = 55
    view_xmin = -130
    view_xmax = -65
    
    PDO_phase = -1
    if PDO_phase < 0:
        PDO_mean = 65.4285714285714
        PDO_stdev = 28.74721058
    elif PDO_phase > 0:
        PDO_mean = 42.20588235
        PDO_stdev = 16.02040254
    
    event_count = int(random.gauss(PDO_mean, PDO_stdev))
    print (event_count)
    for i in range(1,event_count):
        test = Tornado();
        start = test.startpt;
        finish = test.endpt;
        x = [start[1], finish[1]]
        y = [start[0], finish[0]]
        if test.intensity == 3:
            plt.plot(x, y, 'w-', linewidth = 2)
        elif test.intensity == 4:
            plt.plot(x, y, 'k-', linewidth = 2)
        elif test.intensity == 5:
            plt.plot(x, y, 'r-', linewidth = 2)
    zd = test.generate_grid()
    plt.imshow(zd, origin='lower', extent=[view_xmin, view_xmax, view_ymin, view_ymax])
    
    
    plt.show()
