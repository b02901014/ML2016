import sys
import numpy as np

allData = np.genfromtxt('abc.csv',delimiter=',')
print type(allData[1,1])
print np.sort(allData[0,:])

