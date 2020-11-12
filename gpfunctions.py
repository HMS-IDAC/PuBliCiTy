"""
general purpose functions
"""

from os.path import *
from os import listdir, makedirs, remove
import pickle
import shutil
import csv
import time
import matplotlib.pyplot as plt
import tifffile
import os
import numpy as np
from skimage import io as skio
from scipy.ndimage import *
from scipy.signal import convolve
from skimage.morphology import *
from skimage import transform as trfm
from skimage.exposure import equalize_hist, equalize_adapthist, adjust_gamma
from skimage.color import rgb2grey as skimageRGB2Grey
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import cv2


def fileparts(path): # path = file path
    """
    splits file path into components

    *input:*
        file path, e.g. '/path/to/file.ext'

    *output:*
        [root, name, extension], e.g. ['/path/to', 'file', 'ext']
    """

    [p,f] = split(path)
    [n,e] = splitext(f)
    return [p,n,e]

def listfiles(path,token,tokenExclude=None):
    """
    lists files in folder

    *inputs:*
        path: path to folder

        token: a string; filenames containig this are selected

        tokenExcluded: a string; filenames containing this are excluded

    *output:*
        list of file paths
    """

    l = []
    if tokenExclude is None:
        for f in listdir(path):
            fullPath = join(path,f)
            if isfile(fullPath) and token in f:
                l.append(fullPath)
    else:
        for f in listdir(path):
            fullPath = join(path,f)
            if isfile(fullPath) and token in f and not tokenExclude in f:
                l.append(fullPath)
    l.sort()
    return l

def listsubdirs(path,returnFullPath=True):
    """
    lists subdirectories inside directory

    *inputs:*
        path: full path to directory

        returnFullPath: if to return full path, as opposed to subfolders

    *output:*
        list of paths
    """

    l = []
    for f in listdir(path):
        fullPath = join(path,f)
        if isdir(fullPath):
            if returnFullPath:
                l.append(fullPath)
            else:
                l.append(f)
    l.sort()
    return l

def pathjoin(p,ne):
    """
    joins root path with file name (including extension)

    *inputs:*
        p: root path, e.g. '/path/to/folder'

        ne: name+extension, e.g. 'file.ext'

    *output:*
        joined path, e.g. '/path/to/folder/file.ext'

    """

    return join(p,ne)

def saveData(data,path,verbose=False):
    """
    saves python data via pickle

    *inputs:*
        data: data to save

        path: full path where to save

        verbose: True or False; if True, logs terminal message that data is being saved
    """

    if verbose:
        print('saving data')
    dataFile = open(path, 'wb')
    pickle.dump(data, dataFile)

def loadData(path,verbose=False):
    """
    loads python data via pickle

    *inputs:*
        path: full path to data file

        verbose: True or False; if True, logs terminal message that data is being saved

    *output:*
        data

    """

    if verbose:
        print('loading data')
    dataFile = open(path, 'rb')
    return pickle.load(dataFile)

def createFolderIfNonExistent(path):
    """
    creates folder if non existent

    *input:*
        full path to folder
    """

    if not exists(path): # from os.path
        makedirs(path)

def removeFolderIfExistent(path):
    """
    removes folder if existent

    *input*:
        full path to folder
    """

    if exists(path):
        shutil.rmtree(path)

def moveFile(fullPathSource,folderPathDestination):
    """
    moves file with full path *fullPathSource* to folder *folderPathDestination*
    """


    [p,n,e] = fileparts(fullPathSource)
    shutil.move(fullPathSource,pathjoin(folderPathDestination,n+e))

def copyFile(fullPathSource,folderPathDestination):
    """
    copies file with full path *fullPathSource* to folder *folderPathDestination*
    """

    [p,n,e] = fileparts(fullPathSource)
    shutil.copy(fullPathSource,pathjoin(folderPathDestination,n+e))

def removeFile(path):
    """
    removes file with full path *fullPathSource*; no warnings shown
    """

    remove(path)

def writeTable(path,colTitles,matrix):
    """
    simple csv table writing function using Pandas

    *inputs:*
        path: full path where to save csv

        colTitles: list of column titles, e.g. ['col1', 'col2']

        matrix: data to save; number of columns should equal len(colTitles)
    """

    T = {}
    for i in range(len(colTitles)):
        T[colTitles[i]] = matrix[:,i]
    df = pd.DataFrame(T)
    df.to_csv(path, index=False)

def readTable(path):
    """
    simple csv reading function using Pandas

    *input:*
        full path to csv table

    *outputs:*
        [colTitles, matrix], where

        colTitles: list of column titles

        matrix: data
    """

    df = pd.read_csv(path)
    colTitles = df.columns.to_list()
    matrix = df.to_numpy()
    return colTitles, matrix

def tic():
    """
    returns current time
    """

    return time.time()

def toc(t0):
    """
    prints elapsed time given reference time *t0* (obtained via *tic()*)
    """

    print('elapsed time:', time.time()-t0)

def pause(interval):
    """
    pauses process for *interval* seconds
    """

    time.sleep(interval)

def tifread(path):
    """
    reads .tif image file
    """

    return tifffile.imread(path)

def tifwrite(I,path):
    """
    writes image *I* into tif file with fill path *path*
    """

    tifffile.imsave(path, I)

def imshow(I,**kwargs):
    """
    displays image; *kwargs* are passed to matplotlib's *imshow* function
    """

    if not kwargs:
        plt.imshow(I,cmap='gray')
    else:
        plt.imshow(I,**kwargs)
        
    plt.axis('off')
    plt.show()

def imshowlist(L,**kwargs):
    """
    displays list of images, e.g. *L = [I1, I2, I3]*;
    *kwargs* are passed to matplotlib's *imshow* function
    """

    n = len(L)
    for i in range(n):
        plt.subplot(1, n, i+1)
        if not kwargs:
            plt.imshow(L[i],cmap='gray')
        else:
            plt.imshow(L[i],**kwargs)
        plt.axis('off')
    plt.show()

def imread(path):
    """
    generic image reading function; simply wraps skimage.io.imread;
    for .tif images, use *tifread* instead
    """

    return skio.imread(path)

def imwrite(I,path):
    """
    generic image writing function; simply wraps skimage.io.imsave;
    for .tif images, use *tifwrite* instead
    """

    skio.imsave(path,I)

def rgb2gray(I):
    """
    wrap around skimage.color.rgb2grey
    """

    return skimageRGB2Grey(I)

def im2double(I):
    """
    converts uint16, uint8, float32 images into float64 images,
    while normalizing intensity values to range [0, 1]
    """

    if I.dtype == 'uint16':
        return I.astype('float64')/65535
    elif I.dtype == 'uint8':
        return I.astype('float64')/255
    elif I.dtype == 'float32':
        return I.astype('float64')
    elif I.dtype == 'float64':
        return I
    else:
        print('returned original image type: ', I.dtype)
        return I

def imDouble2UInt16(I):
    """
    converts uint16 image to double
    """

    return np.uint16(65535*I)

def imD2U16(I):
    """
    converts uint16 image to double
    """

    return imDouble2UInt16(I)

def size(I):
    """
    returns the size of image *I*, as given by numpy's shape property; literally:
    ::

        return list(I.shape)
    """

    return list(I.shape)

def imresizeDouble(I,sizeOut): # input and output are double
    return trfm.resize(I,(sizeOut[0],sizeOut[1]),mode='reflect')

def imresize3Double(I,sizeOut): # input and output are double
    return trfm.resize(I,(sizeOut[0],sizeOut[1],sizeOut[2]),mode='reflect')

def imresizeUInt8(I,sizeOut): # input and output are UInt8
    return np.uint8(trfm.resize(I.astype(float),(sizeOut[0],sizeOut[1]),mode='reflect',order=0))

def imresize3UInt8(I,sizeOut): # input and output are UInt8
    return np.uint8(trfm.resize(I.astype(float),(sizeOut[0],sizeOut[1],sizeOut[2]),mode='reflect',order=0))

def imresizeUInt16(I,sizeOut): # input and output are UInt16
    return np.uint16(trfm.resize(I.astype(float),(sizeOut[0],sizeOut[1]),mode='reflect',order=0))

def imresize3UInt16(I,sizeOut): # input and output are UInt16
    return np.uint16(trfm.resize(I.astype(float),(sizeOut[0],sizeOut[1],sizeOut[2]),mode='reflect',order=0))

def imresize(I, sizeOut):
    """
    resizes 2D, single channel image (types uint8, uint16, float32 or float64 (double))

    *inputs:*
        I: image

        sizeOut: list of output sizes in rows and columns, e.g. [n_rows, n_cols]

    *output:*
        resized image
    """

    dType = I.dtype
    if dType == 'uint8':
        return imresizeUInt8(I, sizeOut)
    if dType == 'uint16':
        return imresizeUInt16(I, sizeOut)
    if dType == 'double' or dType == 'float32':
        return imresizeDouble(I, sizeOut)

def imresize3(I, sizeOut):
    """
    resizes 3D, single channel image (types uint8, uint16, float32 or float64 (double))

    *inputs:*
        I: image

        sizeOut: list of output sizes in planes, rows and columns, e.g. [n_plns, n_rows, n_cols]

    *output:*
        resized image
    """

    dType = I.dtype
    if dType == 'uint8':
        return imresize3UInt8(I, sizeOut)
    if dType == 'uint16':
        return imresize3UInt16(I, sizeOut)
    if dType == 'double' or dType == 'float32':
        return imresize3Double(I, sizeOut)

def imresize3FromPlanePathList(imPathList, resizeFactor, zStretch, resizeBufferFolderPath):
    """
    resizes 3D images from a list of .tif paths;
    this is a convenience function to help visualizing large 3D image files,
    which are often times saved as a set of planes in a folder;
    2D resizing is aplied for every plane, in parallel, then 3D resizing is applied
    in the stack of planes; thus this does not perform 'true' 3D resizing, which is
    why it's best to use it for visualization purposes, and cases where true 3D would
    be impractical due to memory constraints

    *inputs:*
        imPathList: list of .tif paths

        resizeFactor: float between 0 and 1 indicating amount of resizing

        zStretch: float between 0 and 1 indicating proportion of z stretch
        w.r.t. xy to be performed

        resizeBufferFolderPath: path to folder where resized 2D planes are temporarily saved;
        this folder is created by the script, then removed when processing is finished

    *output:*
        resized 3D image

    *example:*
    ::

        planePathList = listfiles('/path/to/folder', '.tif')
        V = imresize3FromPlanePathList(planePathList, 0.1, 6.17, '/path/to/resize/buffer')
        tifwrite(V, '/path/to/resized/image.tif')
    """

    print('--------------------------------------------------')
    print('getting plane size')
        
    I = tifread(imPathList[0])
    dType = I.dtype
    newSize = list(np.round((resizeFactor*np.array(size(I)))).astype(int))
    nPlanes = len(imPathList)


    print('--------------------------------------------------')
    print('resizing in XY')

    createFolderIfNonExistent(resizeBufferFolderPath)

    # https://scikit-image.org/docs/dev/user_guide/tutorial_parallelization.html
    from joblib import Parallel, delayed
    def task(i):
        I = tifread(imPathList[i])
        I = imresize(I,newSize)
        tifwrite(I,pathjoin(resizeBufferFolderPath, 'I%05d.tif' % i))

    Parallel(n_jobs=4)(delayed(task)(i) for i in range(nPlanes))
    

    print('--------------------------------------------------')
    print('resizing in Z')

    V = np.zeros((nPlanes,newSize[0],newSize[1]), dtype=dType)
    for i in range(nPlanes):
        V[i,:,:] = tifread(pathjoin(resizeBufferFolderPath, 'I%05d.tif' % i))
        
    removeFolderIfExistent(resizeBufferFolderPath)

    return imresize3(V,[int(np.round(nPlanes*resizeFactor*zStretch)), V.shape[1], V.shape[2]])

def imrescale(im,factor): # with respect to center
    im2 = trfm.rescale(im,factor,mode='constant')
    [w1,h1] = im.shape
    [w2,h2] = im2.shape
    r1 = int(h1/2)
    c1 = int(w1/2)
    r2 = int(h2/2)
    c2 = int(w2/2)
    if w2 > w1:
        imout = im2[r2-int(h1/2):r2-int(h1/2)+h1,c2-int(w1/2):c2-int(w1/2)+w1]
    else:
        imout = np.zeros((h1,w1))
        imout[r1-int(h2/2):r1-int(h2/2)+h2,c1-int(w2/2):c1-int(w2/2)+w2] = im2
    return imout

def imadjustgamma(im,gamma): # gamma should be in range (0,1)
    return adjust_gamma(im,gamma)

def imadjustcontrast(im,c): # c should be in the range (0,Inf); c = 1 -> contrast unchanged
    m = np.mean(im)
    return (im-m)*c+m

def normalize(I):
    m = np.min(I)
    M = np.max(I)
    if M > m:
        return (I-m)/(M-m)
    else:
        return I

def snormalize(I):
    m = np.mean(I)
    s = np.std(I)
    if s > 0:
        return (I-m)/s
    else:
        return I

def imadjust(I):
    p1 = np.percentile(I,1)
    p99 = np.percentile(I,99)
    I = (I-p1)/(p99-p1)
    I[I < 0] = 0
    I[I > 1] = 1
    return I

def histeq(I):
    return equalize_hist(I)

def adapthisteq(I):
    return equalize_adapthist(I)

def cat(a,I,J):
    return np.concatenate((I,J),axis=a)

def imtranslate(im,tx,ty): # tx: columns, ty: rows
    tform = trfm.SimilarityTransform(translation = (-tx,-ty))
    return trfm.warp(im,tform,mode='constant')

def imrotate(im,angle): # in degrees, with respect to center
    return trfm.rotate(im,angle)

def imerode(I,r):
    return binary_erosion(I, disk(r))

def imdilate(I,r):
    return binary_dilation(I, disk(r))

def imerode3(I,r):
    return morphology.binary_erosion(I, ball(r))

def imdilate3(I,r):
    return morphology.binary_dilation(I, ball(r))

def sphericalStructuralElement(imShape,fRadius):
    if len(imShape) == 2:
        return disk(fRadius,dtype=float)
    if len(imShape) == 3:
        return ball(fRadius,dtype=float)

def medfilt(I,filterRadius):
    return median_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def maxfilt(I,filterRadius):
    return maximum_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def minfilt(I,filterRadius):
    return minimum_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def ptlfilt(I,percentile,filterRadius):
    return percentile_filter(I,percentile,footprint=sphericalStructuralElement(I.shape,filterRadius))

def imgaussfilt(I,sigma,**kwargs):
    return gaussian_filter(I,sigma,**kwargs)

def imlogfilt(I,sigma,**kwargs):
    return -gaussian_laplace(I,sigma,**kwargs)

def imgradmag(I,sigma): # edge likelihood
    if len(I.shape) == 2:
        dx = imgaussfilt(I,sigma,order=[0,1])
        dy = imgaussfilt(I,sigma,order=[1,0])
        return np.sqrt(dx**2+dy**2)
    if len(I.shape) == 3:
        dx = imgaussfilt(I,sigma,order=[0,0,1])
        dy = imgaussfilt(I,sigma,order=[0,1,0])
        dz = imgaussfilt(I,sigma,order=[1,0,0])
        return np.sqrt(dx**2+dy**2+dz**2)

def localstats(I,radius,justfeatnames=False):
    ptls = [10,30,50,70,90]
    featNames = []
    for i in range(len(ptls)):
        featNames.append('locPtl%d' % ptls[i])
    if justfeatnames == True:
        return featNames
    sI = size(I)
    nFeats = len(ptls)
    F = np.zeros((sI[0],sI[1],nFeats))
    for i in range(nFeats):
        F[:,:,i] = ptlfilt(I,ptls[i],radius)
    return F

def localstats3(I,radius,justfeatnames=False):
    ptls = [10,30,50,70,90]
    featNames = []
    for i in range(len(ptls)):
        featNames.append('locPtl%d' % ptls[i])
    if justfeatnames == True:
        return featNames
    sI = size(I)
    nFeats = len(ptls)
    F = np.zeros((sI[0],sI[1],sI[2],nFeats))
    for i in range(nFeats):
        F[:,:,:,i] = ptlfilt(I,ptls[i],radius)
    return F

def imderivatives(I,sigmas,justfeatnames=False):
    if type(sigmas) is not list:
        sigmas = [sigmas]
    derivPerSigmaFeatNames = ['d0','dx','dy','dxx','dxy','dyy','normGrad','normHessDiag']
    if justfeatnames == True:
        featNames = [];
        for i in range(len(sigmas)):
            for j in range(len(derivPerSigmaFeatNames)):
                featNames.append('derivSigma%d%s' % (sigmas[i],derivPerSigmaFeatNames[j]))
        return featNames
    nDerivativesPerSigma = len(derivPerSigmaFeatNames)
    nDerivatives = len(sigmas)*nDerivativesPerSigma
    sI = size(I)
    D = np.zeros((sI[0],sI[1],nDerivatives))
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        dx = imgaussfilt(I,sigma,order=[0,1])
        dy = imgaussfilt(I,sigma,order=[1,0])
        dxx = imgaussfilt(I,sigma,order=[0,2])
        dyy = imgaussfilt(I,sigma,order=[2,0])
        D[:,:,nDerivativesPerSigma*i  ] = imgaussfilt(I,sigma)
        D[:,:,nDerivativesPerSigma*i+1] = dx
        D[:,:,nDerivativesPerSigma*i+2] = dy
        D[:,:,nDerivativesPerSigma*i+3] = dxx
        D[:,:,nDerivativesPerSigma*i+4] = imgaussfilt(I,sigma,order=[1,1])
        D[:,:,nDerivativesPerSigma*i+5] = dyy
        D[:,:,nDerivativesPerSigma*i+6] = np.sqrt(dx**2+dy**2)
        D[:,:,nDerivativesPerSigma*i+7] = np.sqrt(dxx**2+dyy**2)
    return D
    # derivatives are indexed by the last dimension, which is good for ML features but not for visualization,
    # in which case the expected dimensions are [plane,channel,y(row),x(col)]; to obtain that ordering, do
    # D = np.moveaxis(D,[0,3,1,2],[0,1,2,3])

def imderivatives3(I,sigmas,justfeatnames=False):
    if type(sigmas) is not list:
        sigmas = [sigmas]

    derivPerSigmaFeatNames = ['d0','dx','dy','dz','dxx','dxy','dxz','dyy','dyz','dzz','normGrad','normHessDiag']

    # derivPerSigmaFeatNames = ['d0','normGrad','normHessDiag']

    if justfeatnames == True:
        featNames = [];
        for i in range(len(sigmas)):
            for j in range(len(derivPerSigmaFeatNames)):
                featNames.append('derivSigma%d%s' % (sigmas[i],derivPerSigmaFeatNames[j]))
        return featNames
    nDerivativesPerSigma = len(derivPerSigmaFeatNames)
    nDerivatives = len(sigmas)*nDerivativesPerSigma
    sI = size(I)
    D = np.zeros((sI[0],sI[1],sI[2],nDerivatives)) # plane, channel, y, x
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        dx  = imgaussfilt(I,sigma,order=[0,0,1]) # z, y, x
        dy  = imgaussfilt(I,sigma,order=[0,1,0])
        dz  = imgaussfilt(I,sigma,order=[1,0,0])
        dxx = imgaussfilt(I,sigma,order=[0,0,2])
        dyy = imgaussfilt(I,sigma,order=[0,2,0])
        dzz = imgaussfilt(I,sigma,order=[2,0,0])

        D[:,:,:,nDerivativesPerSigma*i   ] = imgaussfilt(I,sigma)
        D[:,:,:,nDerivativesPerSigma*i+1 ] = dx
        D[:,:,:,nDerivativesPerSigma*i+2 ] = dy
        D[:,:,:,nDerivativesPerSigma*i+3 ] = dz
        D[:,:,:,nDerivativesPerSigma*i+4 ] = dxx
        D[:,:,:,nDerivativesPerSigma*i+5 ] = imgaussfilt(I,sigma,order=[0,1,1])
        D[:,:,:,nDerivativesPerSigma*i+6 ] = imgaussfilt(I,sigma,order=[1,0,1])
        D[:,:,:,nDerivativesPerSigma*i+7 ] = dyy
        D[:,:,:,nDerivativesPerSigma*i+8 ] = imgaussfilt(I,sigma,order=[1,1,0])
        D[:,:,:,nDerivativesPerSigma*i+9 ] = dzz
        D[:,:,:,nDerivativesPerSigma*i+10] = np.sqrt(dx**2+dy**2+dz**2)
        D[:,:,:,nDerivativesPerSigma*i+11] = np.sqrt(dxx**2+dyy**2+dzz**2)

        # D[:,:,:,nDerivativesPerSigma*i   ] = imgaussfilt(I,sigma)
        # D[:,:,:,nDerivativesPerSigma*i+1 ] = np.sqrt(dx**2+dy**2+dz**2)
        # D[:,:,:,nDerivativesPerSigma*i+2 ] = np.sqrt(dxx**2+dyy**2+dzz**2)
    return D
    # derivatives are indexed by the last dimension, which is good for ML features but not for visualization,
    # in which case the expected dimensions are [plane,y(row),x(col)]; to obtain that ordering, do
    # D = np.moveaxis(D,[2,0,1],[0,1,2])

def imfeatures(I=[],sigmaDeriv=1,sigmaLoG=1,locStatsRad=0,justfeatnames=False):
    if type(sigmaDeriv) is not list:
        sigmaDeriv = [sigmaDeriv]
    if type(sigmaLoG) is not list:
        sigmaLoG = [sigmaLoG]
    derivFeatNames = imderivatives([],sigmaDeriv,justfeatnames=True)
    nLoGFeats = len(sigmaLoG)
    locStatsFeatNames = []
    if locStatsRad > 1:
        locStatsFeatNames = localstats([],locStatsRad,justfeatnames=True)
    nLocStatsFeats = len(locStatsFeatNames)
    if justfeatnames == True:
        featNames = derivFeatNames
        for i in range(nLoGFeats):
            featNames.append('logSigma%d' % sigmaLoG[i])
        for i in range(nLocStatsFeats):
            featNames.append(locStatsFeatNames[i])
        return featNames
    nDerivFeats = len(derivFeatNames)
    nFeatures = nDerivFeats+nLoGFeats+nLocStatsFeats
    sI = size(I)
    F = np.zeros((sI[0],sI[1],nFeatures))
    F[:,:,:nDerivFeats] = imderivatives(I,sigmaDeriv)
    for i in range(nLoGFeats):
        F[:,:,nDerivFeats+i] = imlogfilt(I,sigmaLoG[i])
    if locStatsRad > 1:
        F[:,:,nDerivFeats+nLoGFeats:] = localstats(I,locStatsRad)
    return F

def imfeatures3(I=[],sigmaDeriv=2,sigmaLoG=2,sigmaSurf=[],locStatsRad=0,justfeatnames=False):
    if type(sigmaDeriv) is not list:
        sigmaDeriv = [sigmaDeriv]
    if type(sigmaLoG) is not list:
        sigmaLoG = [sigmaLoG]
    if type(sigmaSurf) is not list:
        sigmaSurf = [sigmaSurf]
    derivFeatNames = imderivatives3([],sigmaDeriv,justfeatnames=True)
    nLoGFeats = len(sigmaLoG)
    nSurfFeats = len(sigmaSurf)
    locStatsFeatNames = []
    if locStatsRad > 1:
        locStatsFeatNames = localstats3([],locStatsRad,justfeatnames=True)
    nLocStatsFeats = len(locStatsFeatNames)
    if justfeatnames == True:
        featNames = derivFeatNames
        for i in range(nLoGFeats):
            featNames.append('logSigma%d' % sigmaLoG[i])
        for i in range(nSurfFeats):
            featNames.append('surfSigma%d' % sigmaSurf[i])
        for i in range(nLocStatsFeats):
            featNames.append(locStatsFeatNames[i])
        return featNames
    nDerivFeats = len(derivFeatNames)
    nFeatures = nDerivFeats+nLoGFeats+nSurfFeats+nLocStatsFeats
    sI = size(I)
    F = np.zeros((sI[0],sI[1],sI[2],nFeatures))
    F[:,:,:,:nDerivFeats] = imderivatives3(I,sigmaDeriv)
    for i in range(nLoGFeats):
        F[:,:,:,nDerivFeats+i] = imlogfilt(I,sigmaLoG[i])
    for i in range(nSurfFeats):
        F[:,:,:,nDerivFeats+nLoGFeats+i] = imridgelikl(I,sigmaSurf[i])
    if locStatsRad > 1:
        F[:,:,:,nDerivFeats+nLoGFeats+nSurfFeats:] = localstats3(I,locStatsRad)
    return F

def stack2list(S):
    L = []
    for i in range(size(S)[2]):
        L.append(S[:,:,i])
    return L

def list2stack(l):
    n = len(l)
    nr = l[0].shape[0]
    nc = l[0].shape[1]
    S = np.zeros((n,nr,nc)).astype(l[0].dtype)
    for i in range(len(l)):
        S[i,:,:] = l[i]
    return S

def thrsegment(I,wsBlr,wsThr): # basic threshold segmentation
    G = imgaussfilt(I,sigma=(1-wsBlr)+wsBlr*5) # min 1, max 5
    M = G > wsThr
    return M

def circleKernel(radius,sigma,ftype):
    pi = np.pi
        
    hks = np.max([1,np.ceil(radius+4*sigma)]).astype(int)
    K = np.zeros((int(2*hks+1),int(2*hks+1)))
    K[hks,hks] = 1

    if ftype == 'log':
        K = imlogfilt(K,sigma)
    elif ftype == 'gauss':
        K = imgaussfilt(K,sigma)

    n = np.round(2*pi*radius)

    angles = np.arange(0,2*pi-0.5*pi/n,pi/n)

    S = np.zeros(K.shape);
    for ia in range(len(angles)):
        a = angles[ia]
        v = radius*np.array([np.cos(a), np.sin(a)])
        S = S+imtranslate(K,v[0],v[1])

    if ftype == 'log':
        S = S-np.mean(S)
        sS = np.sqrt(np.sum(np.square(S)))
        S = S/sS
    elif ftype == 'gauss':
        S = normalize(S)

    return S

def morletKernel(stretch,scale,orientation):
    # orientation (in radians)
    theta = -(orientation-90)/360*2*np.pi

    # controls elongation in direction perpendicular to wave
    gamma = 1/(1+stretch)

    # width and height of kernel
    support = np.ceil(2.5*scale/gamma)

    # wavelength (default: 4*scale)
    wavelength = 4*scale

    xmin = -support
    xmax = -xmin
    ymin = xmin
    ymax = xmax

    xdomain = np.arange(xmin,xmax+1)
    ydomain = np.arange(ymin,ymax+1)

    [x,y] = np.meshgrid(xdomain,ydomain)

    xprime = np.cos(theta)*x+np.sin(theta)*y
    yprime = -np.sin(theta)*x+np.cos(theta)*y;

    expf = np.exp(-0.5/np.power(scale,2)*(np.power(xprime,2)+np.power(gamma,2)*np.power(yprime,2)))

    mr = np.multiply(expf,np.cos(2*np.pi/wavelength*xprime))
    mi = np.multiply(expf,np.sin(2*np.pi/wavelength*xprime))

    # mean = 0
    mr = mr-np.sum(mr)/np.prod(mr.shape)
    mi = mi-np.sum(mi)/np.prod(mi.shape)

    # norm = 1
    mr = np.divide(mr,np.sqrt(np.sum(np.power(mr,2))))
    mi = np.divide(mi,np.sqrt(np.sum(np.power(mi,2))))

    return mr, mi

def conv2(I,K,m):
    return convolve(I, K, mode=m) # m = 'full','valid','same'

def conv3(I,K,m):
    return convolve(I, K, mode=m) # m = 'full','valid','same'    

def centerCrop(I,nr,nc):
    nrI = I.shape[0]
    ncI = I.shape[1]
    r0 = int(nrI/2)
    c0 = int(ncI/2)
    nr2 = int(nr/2)
    nc2 = int(nc/2)
    return I[r0-nr2:r0+nr2,c0-nc2:c0+nc2]

def centerCropMultChan(I,nr,nc):
    nrI = I.shape[1]
    ncI = I.shape[2]
    r0 = int(nrI/2)
    c0 = int(ncI/2)
    nr2 = int(nr/2)
    nc2 = int(nc/2)
    return I[:,r0-nr2:r0+nr2,c0-nc2:c0+nc2]

def pad(I,k):
    J = np.zeros((I.shape[0]+2*k,I.shape[1]+2*k))
    J[k:-k,k:-k] = I
    return J

def fullPatchCoordinates2D(nr,nc,patchSize):
    npr = int(np.floor(nr/patchSize)) # number of patch rows
    npc = int(np.floor(nc/patchSize)) # number of patch cols
    fpc = []
    for i in range(npr):
        r0 = i*patchSize
        r1 = r0+patchSize
        for j in range(npc):
            c0 = j*patchSize
            c1 = c0+patchSize
            fpc.append([r0,r1,c0,c1])
    return fpc

def fullPatchCoordinates3D(nz,nr,nc,patchSize):
    npz = int(np.floor(nz/patchSize)) # number of patch plns
    npr = int(np.floor(nr/patchSize)) # number of patch rows
    npc = int(np.floor(nc/patchSize)) # number of patch cols
    fpc = []
    for iZ in range(npz):
        z0 = iZ*patchSize
        z1 = z0+patchSize
        for i in range(npr):
            r0 = i*patchSize
            r1 = r0+patchSize
            for j in range(npc):
                c0 = j*patchSize
                c1 = c0+patchSize
                fpc.append([z0,z1,r0,r1,c0,c1])
    return fpc

def stack2Mosaic(S):
    s = [S.shape[1],S.shape[2]] # '0' assumed to be plane coordinate
    k = int(np.ceil(np.sqrt(S.shape[0])))
    M = np.uint8(np.zeros((k*s[0],k*s[1])))
    for i in range(S.shape[0]):
        r = int(i/k)
        c = i-k*r
        I = np.uint8(255*im2double(S[i,:,:]))
        M[r*s[0]:(r+1)*s[0],c*s[1]:(c+1)*s[1]] = I
    return M

def changeViewPlane(I,currentViewPlane,newViewPlane):
    if currentViewPlane == 'z':
        if newViewPlane == 'x':
            return np.moveaxis(I,[2,0,1],[0,1,2])
        if newViewPlane == 'y':
            return np.moveaxis(I,[1,2,0],[0,1,2])
        else:
            return I
    if currentViewPlane == 'y':
        if newViewPlane == 'z':
            return np.moveaxis(I,[2,0,1],[0,1,2])
        if newViewPlane == 'x':
            return np.moveaxis(I,[1,2,0],[0,1,2])
        else:
            return I
    if currentViewPlane == 'x':
        if newViewPlane == 'y':
            return np.moveaxis(I,[2,0,1],[0,1,2])
        if newViewPlane == 'z':
            return np.moveaxis(I,[1,2,0],[0,1,2])
        else:
            return I

def logKernel3D(sigma):
    d = 4*sigma;
    domain = np.arange(-d,d+1)
    [x,y,z] = np.meshgrid(domain,domain,domain)
    invSigma2 = 1/sigma**2
    K = np.exp(-0.5*invSigma2*(x**2+y**2+z**2))
    dxK = -invSigma2*x*K
    dyK = -invSigma2*y*K
    dzK = -invSigma2*z*K
    dxxK = -invSigma2*(K+x*dxK)
    dyyK = -invSigma2*(K+y*dyK)
    dzzK = -invSigma2*(K+z*dzK)
    L = dxxK+dyyK+dzzK
    L = L-np.mean(L)
    return L

def surfaceKernels(sigma,radius=None,quantity=3):
    if radius is None:
        radius = 4*sigma
    if quantity == 3:
        vs = np.array([[1,0,0],[0,1,0],[0,0,1]])
    elif quantity == 4:
        vs = np.array([[-0.5774,-0.5774,-0.5774],
                       [ 0.5774,-0.5774,-0.5774],
                       [ 0.5774, 0.5774,-0.5774],
                       [-0.5774, 0.5774,-0.5774]])
    elif quantity == 6:
        vs = np.array([[ 0.0000,-0.5257,-0.8507],
                       [ 0.0000,-0.5257, 0.8507],
                       [-0.5257,-0.8507, 0.0000],
                       [-0.5257, 0.8507, 0.0000],
                       [-0.8507, 0.0000,-0.5257],
                       [ 0.8507, 0.0000,-0.5257]])
    elif quantity == 10:
        vs = np.array([[ 0.5774, 0.5774, 0.5774],
                       [ 0.3568, 0.0000, 0.9342],
                       [-0.9342,-0.3568, 0.0000],
                       [-0.5774, 0.5774,-0.5774],
                       [ 0.0000,-0.9342,-0.3568],
                       [-0.9342, 0.3568, 0.0000],
                       [-0.5774, 0.5774, 0.5774],
                       [-0.5774,-0.5774, 0.5774],
                       [ 0.3568, 0.0000,-0.9342],
                       [ 0.0000,-0.9342, 0.3568]])

    L = logKernel3D(sigma)
    Ks = []
    for v in vs:
        r = radius
        d = 2*sigma
        domain = np.arange(-r-d,r+d+1)
        [x,y,z] = np.meshgrid(domain,domain,domain)
        D = np.double(np.abs(v[0]*x+v[1]*y+v[2]*z) < 1)*np.double(np.sqrt(x**2+y**2+z**2) < r)
        K = conv3(D,-L,'same')
        Ks.append(K-np.mean(K))
    return Ks

def imridgelikl(I,sigma): # see imgradmag for 'edge likelihood'
    if len(I.shape) == 2:
        # kx, _ = morletKernel(0,sigma,90)
        # ky, _ = morletKernel(0,sigma,0)
        # Cx = conv2(I,kx,'same')
        # Cy = conv2(I,ky,'same')
        # Cx = Cx*(Cx > 0)
        # Cy = Cy*(Cy > 0)
        # return np.sqrt(np.power(Cx,2)+np.power(Cy,2))

        n = 8
        J = np.zeros((I.shape[0],I.shape[1],n))
        for i in range(n):
            k,_ = morletKernel(1,sigma,i/n*180)
            C = conv2(I,k,'same')
            J[:,:,i] = C*(C > 0)
            
        return np.max(J,axis=2)


    if len(I.shape) == 3:
        # kx, _ = morletKernel(0,sigma,90)
        # Wz = np.zeros(I.shape);
        # for i in range(I.shape[0]):
        #     Wz[i,:,:] = conv2(I[i,:,:],kx,'same')
        # Wz = Wz*(Wz > 0)
        # Vx = changeViewPlane(I,'z','x')
        # Wx = np.zeros(Vx.shape)
        # for i in range(Vx.shape[0]):
        #     Wx[i,:,:] = conv2(Vx[i,:,:],kx,'same')
        # Wx = changeViewPlane(Wx,'x','z')
        # Wx = Wx*(Wx > 0)
        # Vy = changeViewPlane(I,'z','y')
        # Wy = np.zeros(Vy.shape)
        # for i in range(Vy.shape[0]):
        #     Wy[i,:,:] = conv2(Vy[i,:,:],kx,'same')
        # Wy = changeViewPlane(Wy,'y','z')
        # Wy = Wy*(Wy > 0)
        # return np.sqrt(np.power(Wx,2)+np.power(Wy,2)+np.power(Wz,2))

        # Wz = np.zeros(I.shape);
        # for i in range(I.shape[0]):
        #     print(i)
        #     Wz[i,:,:] = imridgelikl(I[i,:,:],sigma)
        # Vx = changeViewPlane(I,'z','x')
        # Wx = np.zeros(Vx.shape)
        # for i in range(Vx.shape[0]):
        #     print(i)
        #     Wx[i,:,:] = imridgelikl(Vx[i,:,:],sigma)
        # Wx = changeViewPlane(Wx,'x','z')
        # Vy = changeViewPlane(I,'z','y')
        # Wy = np.zeros(Vy.shape)
        # for i in range(Vy.shape[0]):
        #     print(i)
        #     Wy[i,:,:] = imridgelikl(Vy[i,:,:],sigma)
        # Wy = changeViewPlane(Wy,'y','z')
        # W = np.zeros((I.shape[0],I.shape[1],I.shape[2],3))
        # W[:,:,:,0] = Wx
        # W[:,:,:,1] = Wy
        # W[:,:,:,2] = Wz
        # return np.max(W,axis=3)
        
        Ks = surfaceKernels(sigma,quantity=10)
        W = np.zeros((I.shape[0],I.shape[1],I.shape[2],len(Ks)))
        for i in range(len(Ks)):
            C = conv3(I,Ks[i],'same')
            W[:,:,:,i] = C*(C > 0)
        return np.max(W,axis=3)

def findSpots2D(I,sigma,thr=0.5):
    # I: double, range [0,1] image
    # sigma: sigma of spot (should be integer 1,2,3,...)
    # thr: correlation selection threshold
    sigma = int(sigma)

    J = imlogfilt(I,sigma)
    K = peak_local_max(J,indices=False,min_distance=sigma)
    K = K*(J > 0)
    P = regionprops(label(K))

    I3 = np.zeros((I.shape[0],I.shape[1],3))
    for i in range(3):
        I3[:,:,i] = I

    psList = []
    for i in range(len(P)):
        row, col = P[i].centroid
        row = int(row)
        col = int(col)
        if P[i].area == 1:
            psList.append([row,col])
            I3[row,col,:] = 0
            I3[row,col,0] = 1
        else:
            I3[row,col,:] = 0
            I3[row,col,2] = 1

    hks = np.round(2*sigma)
    k = np.zeros((2*hks+1,2*hks+1))
    k[hks,hks] = 1
    k = imgaussfilt(k,sigma)
    flatK = k.flatten()

    sPsList = []
    for i in range(len(psList)):
        [row,col] = psList[i]
        r0 = row-hks
        c0 = col-hks
        r1 = row+hks+1
        c1 = col+hks+1
        if r0 >= 0 and r1 <= I.shape[0] and c0 >= 0 and c1 <= I.shape[1]:
            C = I[r0:r1,c0:c1]
            if np.corrcoef(flatK,C.flatten())[1,0] > thr:
                sPsList.append(psList[i])
                I3[row, col, 1] = 1

    return psList, sPsList, I3

def findSpots3D(I,sigma,thr=0.5):
    # I: double, range [0,1] image
    # sigma: sigma of spot (should be integer 1,2,3,...)
    # thr: correlation selection threshold
    sigma = int(sigma)

    J = imlogfilt(I,sigma)
    K = peak_local_max(J,indices=False,min_distance=sigma)
    K = K*(J > 0)
    P = regionprops(label(K))

    psList = []
    for i in range(len(P)):
        pln, row, col = P[i].centroid
        pln = int(pln)
        row = int(row)
        col = int(col)
        if P[i].area == 1:
            psList.append([pln, row, col])

    hks = np.round(2 * sigma)
    ks = 2*hks+1
    k = np.zeros((ks,ks,ks))
    k[hks, hks, hks] = 1
    k = imgaussfilt(k, sigma)
    flatK = k.flatten()

    sPsList = []
    S = np.zeros(I.shape,dtype=bool)
    for i in range(len(psList)):
        [pln, row, col] = psList[i]
        p0 = pln - hks
        r0 = row - hks
        c0 = col - hks
        p1 = pln + hks + 1
        r1 = row + hks + 1
        c1 = col + hks + 1
        if p0 >= 0 and p1 <= I.shape[0] and r0 >= 0 and r1 <= I.shape[1] and c0 >= 0 and c1 <= I.shape[2]:
            C = I[p0:p1, r0:r1, c0:c1]
            if np.corrcoef(flatK, C.flatten())[1, 0] > thr:
                sPsList.append(psList[i])
                S[pln, row, col] = True

    return psList, sPsList, S

def planes2rgb(R=None,G=None,B=None):
    if R is not None:
        nr,nc = R.shape
        dtype = R.dtype
    elif G is not None:
        nr,nc = G.shape
        dtype = G.dtype
    elif B is not None:
        nr,nc = B.shape
        dtype = B.dtype

    RGB = np.zeros((nr,nc,3)).astype(dtype)
    if R is not None:
        RGB[:,:,0] = R
    if G is not None:
        RGB[:,:,1] = G
    if B is not None:
        RGB[:,:,2] = B

    return RGB

def imbinarize(I):
    return np.double(I > threshold_otsu(I))

def bwInterpSingleObjectMasks(I, J, a):
    """
    interpolates two object masks

    *inputs:*
        I, J: masks
        
        a: interpolation factor, between 0 and 1;
        0 returns I, 1 return J

    *output:*
        interpolated mask

    *example:*
    ::

        X,Y = np.meshgrid(np.arange(400),np.arange(400))
        I = np.sqrt((X-150)**2+(Y-210)**2) < 50
        J = np.sqrt(0.5*(X-200)**2+(Y-200)**2) < 100
        a = 0.5
        K = bwInterpSingleObjectMasks(I, J, a)
        imshowlist([I, K, J])
    """

    I = np.uint8(I)
    J = np.uint8(J)

    _, contoursI, hierarchyI = cv2.findContours(I, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contoursJ, hierarchyJ = cv2.findContours(J, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contI0 = np.squeeze(contoursI[0])
    contJ0 = np.squeeze(contoursJ[0])

    def closestPoints(c0, c1):
        i_c0 = 0
        j_c1 = 0
        min_dist = np.inf
        for i in range(c0.shape[0]):
            for j in range(c1.shape[0]):
                d = np.sqrt(np.sum((c0[i,:]-c1[j,:])**2))
                if d < min_dist:
                    min_dist = d
                    i_c0 = i
                    j_c1 = j
        return i_c0, j_c1

    i, j = closestPoints(contI0, contJ0)

    nI = contI0.shape[0]
    nJ = contJ0.shape[0]
    shift_i = np.roll(np.arange(nI),-i)
    shift_j = np.roll(np.arange(nJ),-j)

    contI0 = contI0[shift_i,:]
    contJ0 = contJ0[shift_j,:]

    if nI > nJ:
        contJ1 = np.zeros(contI0.shape, dtype=contI0.dtype)
        for i in range(nI):
            contJ1[i,:] = contJ0[int(np.floor(i/nI*nJ)),:]
        contJ0 = contJ1
    else:
        contI1 = np.zeros(contJ0.shape, dtype=contJ0.dtype)
        for j in range(nJ):
            contI1[j,:] = contI0[int(np.floor(j/nJ*nI)),:]
        contI0 = contI1

    contK0 = np.round((1-a)*contI0+a*contJ0).astype(int)

    K = cv2.fillPoly(np.zeros(I.shape, dtype=np.uint8), pts=[contK0], color=1)
    return K > 0

def interpolateAnnotations3D(A, classIdx):
    A2 = np.copy(A)

    i_list = []
    for i in range(A.shape[0]):
        if np.any(A[i,:,:] == classIdx):
            i_list.append(i)

    n_intervals = len(i_list)-1
    for ii in range(n_intervals):
        print('interpolating interval %d of %d for class %d' % (ii+1, n_intervals, classIdx))
        i0 = i_list[ii]
        i1 = i_list[ii+1]
        M0 = A[i0,:,:] == classIdx
        M1 = A[i1,:,:] == classIdx
        for j in range(i0,i1+1):
            alpha = (j-i0)/(i1-i0)
            M = bwInterpSingleObjectMasks(M0, M1, alpha)
            A2[j,:,:] = classIdx*np.uint8(M)

    return A2
