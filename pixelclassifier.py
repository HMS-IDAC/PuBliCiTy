"""
2D semantic segmentation
"""

from gpfunctions import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time

def parseLabelFolder(trainPath):
    imPathList = listfiles(trainPath,'_Img.tif')

    imPath = imPathList[0]
    [p,n,e] = fileparts(imPath)
    lPath = pathjoin(p,'%s_Ant.tif' % n[:-4])
    Li = tifread(lPath)
    nClasses = np.max(Li)

    oneClassProblem = nClasses == 1
    if oneClassProblem:
        nClasses = 2 # we'll create 'background' class with random pixels in the complementary of class 1

    nSamples = 0
    imList = []
    lbList = []
    for imPath in imPathList:
        [p,n,e] = fileparts(imPath)
        imList.append(im2double(tifread(imPath)))
        L = []
        lPath = pathjoin(p,'%s_Ant.tif' % n[:-4])
        Li = tifread(lPath)

        if oneClassProblem:
            nLi1 = np.sum(Li > 0)
            Li2Mask = (np.random.rand(Li.shape[0],Li.shape[1]) < nLi1/(np.prod(Li.shape)-nLi1))*(Li == 0) > 0

            # Li2Mask = imdilate(Li == 1,5)*(1-imdilate(Li == 1,3)) > 0

            Li[Li2Mask] = 2
            # print(sum(Li == 1),sum(Li == 2))
            # imshowlist([Li == 1,Li == 2])
        else: # class balance
            nLi = np.zeros(nClasses)
            for iClass in range(nClasses):
                nLi[iClass] = np.sum(Li == iClass+1)
            
            minNLi, iMinNLi = np.min(nLi), np.argmin(nLi)

            for iClass in range(nClasses):
                if iClass != iMinNLi:
                    Mask0 = Li == iClass+1
                    Mask1 = (np.random.rand(Li.shape[0],Li.shape[1]) < minNLi/nLi[iClass])*Mask0 > 0
                    Li[Mask0] = 0
                    Li[Mask1] = iClass+1
                    
                # print(np.sum(Li == iClass+1))

        for iClass in range(nClasses):
            Lii = Li == iClass+1
            L.append(Lii)
            nSamples += np.sum(Lii > 0)
        lbList.append(L)

    return nClasses, nSamples, imList, lbList

def setupTraining(nClasses,nSamples,imList,lbList,sigmaDeriv=2,sigmaLoG=[],locStatsRad=0):
    featNames = imfeatures(sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,locStatsRad=locStatsRad,justfeatnames=True)
    nFeatures = len(featNames)
    X = np.zeros((nSamples,nFeatures))
    Y = np.zeros((nSamples))
    i0 = 0
    for iImage in range(len(imList)):
        I = imList[iImage]
        F = imfeatures(I=I,sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,locStatsRad=locStatsRad)
        for iClass in range(nClasses):
            Li = lbList[iImage][iClass]
            indices = Li > 0
            l = np.sum(indices)
            x = np.zeros((l,nFeatures))
            for iFeat in range(nFeatures):
                Fi = F[:,:,iFeat]
                xi = Fi[indices]
                x[:,iFeat] = xi
            y = (iClass+1)*np.ones((l))
            X[i0:i0+l,:] = x
            Y[i0:i0+l] = y
            i0 = i0+l
    return X, Y, {'nClasses': nClasses,
                  'nFeatures': nFeatures,
                  'featNames': featNames,
                  'sigmaDeriv': sigmaDeriv,
                  'sigmaLoG': sigmaLoG,
                  'locStatsRad': locStatsRad}

def rfcTrain(X,Y,params):
    rfc = RandomForestClassifier(n_estimators=100,n_jobs=-1,min_samples_leaf=50)
    rfc = rfc.fit(X, Y)
    model = params
    model['rfc'] = rfc
    model['featImport'] = rfc.feature_importances_
    return model

def train(trainPath,**kwargs):
    print('training...')
    startTime = time.time()
    nClasses, nSamples, imList, lbList = parseLabelFolder(trainPath)
    X, Y, params = setupTraining(nClasses,nSamples,imList,lbList,**kwargs)
    model = rfcTrain(X,Y,params)
    print('...done')
    print('elapsed time: ', time.time()-startTime)
    return model

def classify(I,model,output='classes'):
    print('classifying...')
    startTime = time.time()
    rfc = model['rfc']
    F = imfeatures(I=I,sigmaDeriv=model['sigmaDeriv'],sigmaLoG=model['sigmaLoG'],locStatsRad=model['locStatsRad'])
    sI = size(I)
    M = np.zeros((sI[0],sI[1],model['nClasses']))
    if output == 'classes':
        out = rfc.predict(F.reshape(-1,model['nFeatures']))
        C = out.reshape(I.shape)
        for i in range(model['nClasses']):
            M[:,:,i] = C == i+1
    elif output == 'probmaps':
        out = rfc.predict_proba(F.reshape(-1,model['nFeatures']))
        for i in range(model['nClasses']):
            M[:,:,i] = out[:,i].reshape(I.shape)
    print('...done')
    print('elapsed time: ', time.time()-startTime)
    return M

def plotFeatImport(fi,fn):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 5)
    y_pos = range(len(fn))
    ax.barh(y_pos, fi)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fn)
    ax.invert_yaxis()
    ax.set_title('Feature Importance')
    plt.show()

if __name__ == "__main__":
    from gpfunctions import *
    import os
    
    # -------------------------
    # train

    trainPath = os.path.abspath('DataForPC/Train')
    model = train(trainPath,sigmaDeriv=[2,4,8,16],sigmaLoG=[2,4,8,16])

    plotFeatImport(model['featImport'],model['featNames'])

    # -------------------------
    # segment

    path = os.path.abspath('DataForPC/Train/I00000_Img.tif')
    I = im2double(tifread(path))

    C = classify(I,model,output='classes')
    P = classify(I,model,output='probmaps')

    imshowlist([I]+stack2list(C)+stack2list(P))