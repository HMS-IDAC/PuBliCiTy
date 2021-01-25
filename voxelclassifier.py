"""
3D semantic segmentation

*demo:*
::

    import voxelclassifier as vc
    from gpfunctions import *
    import os

    trainPath = os.path.abspath('DataForVC/Train')
    model = vc.train(trainPath,sigmaDeriv=[2],sigmaLoG=[],locStatsRad=0)
    vc.plotFeatImport(model['featImport'],model['featNames'])

    path = os.path.abspath('DataForVC/Train/Mitochondria_Img.tif')
    I = im2double(tifread(path))
    P = vc.classify(I,model,output='probmaps')

    pln = 150
    imshowlist([I[pln,:,:], P[pln,:,:,0], P[pln,:,:,1]])
"""

from gpfunctions import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time

def parseLabelFolder(trainPath):
    """
    parses a folder of training data to return images, labels and metadata

    *input:*
        trainPath: full path to folder containing training images and labels;
        note that each pair (image, label) should have the same name except for
        the last 8 characters, which should be '_Img.tif' and '_Ant.tif', respectively;
        a label is assumed to be an image of same size as the pairing image, of type
        uint8, where class 1 corresponds to voxels of intensity 1, class 2 to voxels
        of intensity 2, and so on; if there's only one class, the function
        assumes the complement is class 2, and randomly samples it so that the number
        of voxels from class 1 and 2 are the same withing a label (annotation) image

    *outputs:*
        nClasses: number of classes

        nSamples: total number of annotated voxels

        imList: list of images

        lbList: list of labels
    """

    imPathList = listfiles(trainPath,'_Img.tif')

    imPath = imPathList[0]
    [p,n,e] = fileparts(imPath)
    lPath = pathjoin(p,'%s_Ant.tif' % n[:-4])
    Li = tifread(lPath)
    nClasses = np.max(Li)
    
    oneClassProblem = nClasses == 1
    if oneClassProblem:
        nClasses = 2 # we'll create 'background' class with random voxels in the complementary of class 1

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
            Li2Mask = (np.random.rand(Li.shape[0],Li.shape[1],Li.shape[2]) < nLi1/(np.prod(Li.shape)-nLi1))*(Li == 0) > 0
            Li[Li2Mask] = 2

            # print(sum(Li == 1),sum(Li == 2))
        else: # class balance
            nLi = np.zeros(nClasses)
            for iClass in range(nClasses):
                nLi[iClass] = np.sum(Li == iClass+1)
            
            minNLi, iMinNLi = np.min(nLi), np.argmin(nLi)

            if minNLi > 0:
                for iClass in range(nClasses):
                    if iClass != iMinNLi:
                        Mask0 = Li == iClass+1
                        Mask1 = (np.random.rand(Li.shape[0],Li.shape[1],Li.shape[2]) < minNLi/nLi[iClass])*Mask0 > 0
                        Li[Mask0] = 0
                        Li[Mask1] = iClass+1


        for iClass in range(nClasses):
            Lii = Li == iClass+1
            L.append(Lii)
            nSamples += np.sum(Lii > 0)
        lbList.append(L)

    return nClasses, nSamples, imList, lbList

def setupTraining(nClasses,nSamples,imList,lbList,sigmaDeriv=2,sigmaLoG=[],sigmaSurf=[],locStatsRad=0):
    """
    re-formats training data for training

    *inputs:*
        nClasses: number of classes

        nSamples: number of annotated voxels

        imList: list of images

        lbList: list of labels (annotations)

        sigmaDeriv: sigma, or list of sigmas, for derivative features (computed via gpfunctions.imderivatives3)

        sigmaLoG: sigma, or list of sigmas, for laplacian-of-gaussian features (computed via gpfunctions.imlogfilt)

        sigmaSurf: sigma, or list of sigmas, for surface features (computed via gpfunctions.imridgelikl)

        locStatRad: radius for local percentiles (computed via gpfunctions.localstats3)

    *outputs:*
        X: a matrix of nSamples x nFeatures containing image features

        Y: a matrix of nSamples x 1 containing labels

        metaDataDict: a dictionary with the following keys: nClasses, nFeatures (number of features),
        featNames (list of feature names), sigmaDeriv, sigmaLoG, sigmaSurf, locStatRad
    """

    featNames = imfeatures3(sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,sigmaSurf=sigmaSurf,locStatsRad=locStatsRad,justfeatnames=True)
    nFeatures = len(featNames)
    X = np.zeros((nSamples,nFeatures))
    Y = np.zeros((nSamples))
    i0 = 0
    for iImage in range(len(imList)):
        I = imList[iImage]
        F = imfeatures3(I=I,sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,sigmaSurf=sigmaSurf,locStatsRad=locStatsRad)
        for iClass in range(nClasses):
            Li = lbList[iImage][iClass]
            indices = Li > 0
            l = np.sum(indices)
            x = np.zeros((l,nFeatures))
            for iFeat in range(nFeatures):
                Fi = F[:,:,:,iFeat]
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
                  'sigmaSurf': sigmaSurf,
                  'locStatsRad': locStatsRad}

def rfcTrain(X,Y,params):
    """
    random forest classifier trainer

    *inputs:*
        X: a matrix of nSamples x nFeatures containing image features

        Y: a matrix of nSamples x 1 containing labels

        params: a dictionary of parameters, as in metaDataDict returned by setupTraining

    *output:*
        model: a dictionary containing the keys, values from params, in addicion to
        key 'rfc' paired with the random forest models, and key 'featImport' paired
        with feature importances (which can be used in plotFeatImport)
    """

    rfc = RandomForestClassifier(n_estimators=100,n_jobs=-1,min_samples_leaf=50)
    rfc = rfc.fit(X, Y)
    model = params
    model['rfc'] = rfc
    model['featImport'] = rfc.feature_importances_
    return model

def train(trainPath,**kwargs):
    """
    classifier training function

    *inputs:*
        trainPath: path to folder contining images and labels (annotations)

        kwargs: extra optional parameters passed to setupTraining: sigmaDeriv, sigmaLoG, sigmaSurf, locStatsRad

    *output:*
        model: same as the output of rfcTrain
    """

    print('training...')
    startTime = time.time()
    nClasses, nSamples, imList, lbList = parseLabelFolder(trainPath)
    X, Y, params = setupTraining(nClasses,nSamples,imList,lbList,**kwargs)
    model = rfcTrain(X,Y,params)
    print('...done')
    print('elapsed time: ', time.time()-startTime)
    return model

def classify(I,model,output='classes'):
    """
    classifies voxels given an image and a trained model

    *inputs:*
        I: image

        model: trained model

        output: either 'classes' or 'probmaps'; if 'classes', the output is a stack of volumes
        containing masks for each class; if 'probmaps', the output is a stack of volumes containing
        probabilities (numbers between 0 and 1) for voxels to belong to that class

    *output:*
        stack of masks or probability maps, depending on if output='classes' or output='probmaps'
    """

    print('classifying...')
    startTime = time.time()
    rfc = model['rfc']
    F = imfeatures3(I=I,sigmaDeriv=model['sigmaDeriv'],sigmaLoG=model['sigmaLoG'],sigmaSurf=model['sigmaSurf'],locStatsRad=model['locStatsRad'])
    sI = size(I)
    M = np.zeros((sI[0],sI[1],sI[2],model['nClasses']))
    # M = np.zeros((sI[0],model['nClasses'],sI[1],sI[2]))
    if output == 'classes':
        out = rfc.predict(F.reshape(-1,model['nFeatures']))
        C = out.reshape(I.shape)
        for i in range(model['nClasses']):
            M[:,:,:,i] = C == i+1
            # M[:,i,:,:] = C == i+1
    elif output == 'probmaps':
        out = rfc.predict_proba(F.reshape(-1,model['nFeatures']))
        for i in range(model['nClasses']):
            M[:,:,:,i] = out[:,i].reshape(I.shape)
            # M[:,i,:,:] = out[:,i].reshape(I.shape)
    print('...done')
    print('elapsed time: ', time.time()-startTime)
    return M

def plotFeatImport(fi,fn):
    """
    plots feature importances

    *inputs:*
        fi: list of feature importances; these can be accessed from the model (output of train)
        via model['featImport']

        fn: list of feature names; these can be accessed from the model (output of train)
        via model['featNames']

    *output:*
        (none; a plot will be displayed)
    """

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

    trainPath = os.path.abspath('DataForVC')
    model = train(trainPath,sigmaDeriv=[2],sigmaLoG=[],locStatsRad=0)
    plotFeatImport(model['featImport'],model['featNames'])

    # -------------------------
    # segment

    path = os.path.abspath('DataForVC/Drosophila_Img.tif')
    I = im2double(tifread(path))
    P = classify(I,model,output='probmaps')

    tifwrite(np.uint8(255*P[:,:,:,1]),'P1.tif')