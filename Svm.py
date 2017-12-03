# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:24:21 2017

@author: Q
"""
import numpy as np

def loadDataSet(fileName):  
    dataMat = [] 
    labelMat = []
    with open(fileName) as f:
        for line in f.readlines():
            line = line.strip().split()
            dataMat.append([float(line[0]),float(line[1])])
            labelMat.append(int(line[2]))
    return dataMat,labelMat
    
def selectJrand(i,m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iters = 0
    while (iters < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fxi - float(labelMat[i])
            if( ((labelMat[i]*Ei)<-toler) and (alphas[i]<C) ) or ((labelMat[i]*Ei>toler) and(alphas[i] > 0)):
                j = selectJrand(i,m)
                fxj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C + alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H:
                    print('L=H')
                    continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print('J is not move')
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1 = b -Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b -Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i])and(C>alphas[i]):
                    b = b1
                elif(0<alphas[j])and(C>alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print('iter:%d   i:%d   pairs:%d'%(iters,i,alphaPairsChanged))
        if(alphaPairsChanged == 0):
            iters += 1
        else:
            iters = 0
        print('iteration number :%d'%iters)
    return b,alphas
if __name__ == '__main__':
    data,label = loadDataSet('testSet.txt')
    b,alpha = smoSimple(data,label,0.6,0.001,40)
    print(b,alpha)
                        
                
                
                
                
                
                
                
                
                
                
                
                
                
            