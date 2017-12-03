# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:14:21 2017

@author: Q
"""

import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt

def loadDataSet(fileName):  
    dataMat = [] 
    labelMat = []
    with open(fileName) as f:
        for line in f.readlines():
            line = line.strip().split()
            dataMat.append([float(line[0]),float(line[1])])
            labelMat.append(round(float(line[2])))
    return dataMat,labelMat
    
data,label = loadDataSet('testSetRBF.txt')
data = np.array(data)
label = np.array(label)
clf = svm.SVC()
clf.fit(data,label)


fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.scatter(data[:,0],data[:,1],c=label, s=30, cmap=plt.cm.Paired)


xx = ax.get_xlim()
yy = ax.get_ylim()
XX = np.linspace(xx[0],xx[1],30)
YY = np.linspace(yy[0],yy[1],30)
xxx, yyy = np.meshgrid(XX,YY)
xy = np.dstack((xxx,yyy)).reshape(np.size(xxx),2)
xy_labels = clf.decision_function(xy).reshape(xxx.shape)
ax.contour(xxx,yyy,xy_labels,colors = 'b',levels = [-1,0,1],linestyles=['--','-','--'])


plt.show()





