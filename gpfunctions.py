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
from numpy.matlib import repmat
from skimage import io as skio
from scipy.ndimage import *
from scipy.signal import convolve
from scipy.ndimage.morphology import binary_fill_holes
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

def imrescale(im,factor):
    """
    rescales image with respect to center

    *inputs:*
        im: image

        factor: rescale factor (float)

    *output:*
        rescaled image
    """

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
    """
    gamma correction of luminance values

    *inputs:*
        im: image

        gamma: gamma factor, a float between 0 and 1

    *output:*
        corrected image
    """

    return adjust_gamma(im,gamma)

def imadjustcontrast(im,c): # c should be in the range (0,Inf); c = 1 -> contrast unchanged
    """
    simple contrast adjustment, keeping average pixel intensity

    *inputs:*
        im: image

        c: contrast adjustment factor; should be in the range (0, Inf);
        c = 1 implies no contrast change

    *output:*
        contrast-adjusted image

    *implementation:*
    ::

        m = np.mean(im)
        return (im-m)*c+m        
    """

    m = np.mean(im)
    return (im-m)*c+m

def normalize(I):
    """
    linearly maps pixel intensity to range [0, 1], unless the original range is 0
    (i.e. all pixel values are the same), in which case output = input

    *input:*
        image

    *output:*
        normalized image
    """

    m = np.min(I)
    M = np.max(I)
    if M > m:
        return (I-m)/(M-m)
    else:
        return I

def snormalize(I):
    """
    linearly changes pixel intensities so that the output pixel intensities
    have average 0 and standard deviation 1;
    if original standard deviation is 0, output = input

    *input:*
        image

    *output:*
        normalized image
    """

    m = np.mean(I)
    s = np.std(I)
    if s > 0:
        return (I-m)/s
    else:
        return I

def imadjust(I):
    """
    maps 1st pixel intensity percentile to 0 and 99th to 1, linearly stretching
    intensities in between, and clipping intensities outside the interval

    *input:*
        image

    *output:*
        adjusted image

    *implementation:*
    ::

        p1 = np.percentile(I,1)
        p99 = np.percentile(I,99)
        I = (I-p1)/(p99-p1)
        I[I < 0] = 0
        I[I > 1] = 1
        return I    
    """

    p1 = np.percentile(I,1)
    p99 = np.percentile(I,99)
    I = (I-p1)/(p99-p1)
    I[I < 0] = 0
    I[I > 1] = 1
    return I

def histeq(I):
    """
    returns histogram-equalized input
    """

    return equalize_hist(I)

def adapthisteq(I):
    """
    returns adaptive-histogram-equalized input
    """

    return equalize_adapthist(I)

def cat(a,I,J):
    """
    concatenates *I* and *J* along axis *a*

    *implementation:*
    ::

        return np.concatenate((I,J),axis=a)        
    """

    return np.concatenate((I,J),axis=a)

def imtranslate(im,tx,ty): # tx: columns, ty: rows
    """
    applies translation transform

    *inputs:*
        im: image

        tx: translation in x (columns)

        ty: translation in y (rows)

    *output:*
        translated image
    """

    tform = trfm.SimilarityTransform(translation = (-tx,-ty))
    return trfm.warp(im,tform,mode='constant')

def imrotate(im,angle):
    """
    rotates image with respect to center

    *inputs:*
        im: image

        angle: angle of rotation, in degrees

    *output:*
        rotated image
    """

    return trfm.rotate(im,angle)

def imfillholes(I):
    """
    binary fill holes

    *inputs*:
        I: binary image

    *output:*
        binary filled image
    """

    return binary_fill_holes(I)

def imerode(I,r):
    """
    binary erosion

    *inputs*:
        I: binary image

        r: radius of disk structuring element

    *output:*
        eroded image
    """

    return binary_erosion(I, disk(r))

def imdilate(I,r):
    """
    binary dilation

    *inputs*:
        I: binary image

        r: radius of disk structuring element

    *output:*
        dilated image
    """
    return binary_dilation(I, disk(r))

def imerode3(I,r):
    """
    binary 3D erosion

    *inputs*:
        I: binary 3D image

        r: radius of sphere structuring element

    *output:*
        eroded 3D image
    """

    return morphology.binary_erosion(I, ball(r))

def imdilate3(I,r):
    """
    binary 3D dilation

    *inputs*:
        I: binary 3D image

        r: radius of sphere structuring element

    *output:*
        dilated 3D image
    """

    return morphology.binary_dilation(I, ball(r))

def sphericalStructuralElement(imShape,fRadius):
    if len(imShape) == 2:
        return disk(fRadius,dtype=float)
    if len(imShape) == 3:
        return ball(fRadius,dtype=float)

def medfilt(I,filterRadius):
    """
    median filter using spherical structural element (disk for 2D images and sphere for 3D)

    *inputs:*
        I: image

        filterRadius: radius of median filter

    *output:*
        filtered image
    """

    return median_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def maxfilt(I,filterRadius):
    """
    max filter using spherical structural element (disk for 2D images and sphere for 3D)

    *inputs:*
        I: image

        filterRadius: radius of max filter

    *output:*
        filtered image
    """

    return maximum_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def minfilt(I,filterRadius):
    """
    min filter using spherical structural element (disk for 2D images and sphere for 3D)

    *inputs:*
        I: image

        filterRadius: radius of min filter

    *output:*
        filtered image
    """

    return minimum_filter(I,footprint=sphericalStructuralElement(I.shape,filterRadius))

def ptlfilt(I,percentile,filterRadius):
    """
    percentile filter using spherical structural element (disk for 2D images and sphere for 3D);
    note that ptlfilt(I, 50, filterRadius) = medfilt(I, filterRadius)

    *inputs:*
        I: image
        
        percentile: percentile value between 0 and 100

        filterRadius: radius of percentile filter

    *output:*
        filtered image
    """
    return percentile_filter(I,percentile,footprint=sphericalStructuralElement(I.shape,filterRadius))

def imgaussfilt(I,sigma,**kwargs):
    """
    gaussian (blur) filter

    *inputs:*
        I: image

        sigma: sigma parameter of gaussian filter

        kwargs: extra arguments passed to scipy.ndimage.gaussian_filter

    *output:*
        filtered image   
    """

    return gaussian_filter(I,sigma,**kwargs)

def imlogfilt(I,sigma,**kwargs):
    """
    laplacian of gaussian (LoG) filter

    *inputs:*
        I: image

        sigma: sigma parameter of LoG filter

        kwargs: extra arguments passed to scipy.ndimage.gaussian_laplace

    *output:*
        filtered image; technically the inverse of the LoG operator, so that bright spots
        in the image result in bright spots in the output
    """    

    return -gaussian_laplace(I,sigma,**kwargs)

def imgradmag(I,sigma): # edge likelihood
    """
    gradient magnitude, i.e., norm of the gradient image

    *inputs:*
        I: 2D or 3D image

        sigma: sigma for filtering before computing the gradient

    *output:*
        gradient magnitude image
    """

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
    """
    computes local percentiles in 2D images

    *input:*
        I: 2D image

        radius: radius of disk structuring element

        justfeatnames: True or False; if to output just feature names
        (this is useful to allocate memory in certain machine learning tasks)

    *output:*
        stack of percentile filtered images, for percentiles 10, 30, 50, 70, 90,
        if justfeatnames = False, otherwise just feature names

    """

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
    """
    computes local percentiles in 3D images

    *input:*
        I: 3D image

        radius: radius of sphere structuring element

        justfeatnames: True or False; if to output just feature names
        (this is useful to allocate memory in certain machine learning tasks)

    *output:*
        stack of percentile filtered images, for percentiles 10, 30, 50, 70, 90,
        if justfeatnames = False, otherwise just feature names

    """

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
    """
    computes stack of derivatives of 2D image, up to second order

    *inputs:*
        I: 2D image

        sigmas: list of sigmas for gaussian filtering before computing derivatives

        justfeatnames: True or False; if to output just feature names
        (this is useful to allocate memory in certain machine learning tasks)

    *output:*
        stack of derivatives if justfeatnames = False, otherwise just feature names;
        for each sigma, the following derivatives are computed: d0, dx, dy, dxx, dxy, dyy,
        normGrad (gradient magnitude), normHessDiag (norm of diagonal of hessian)

        derivatives are indexed by the last dimension,
        which is good for ML features but not for visualization,
        in which case the expected dimensions are [plane,channel,y(row),x(col)];
        to obtain that ordering, do D = np.moveaxis(D,[0,3,1,2],[0,1,2,3]) on the output D
    """

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

def imderivatives3(I,sigmas,justfeatnames=False):
    """
    computes stack of derivatives of 3D image, up to second order

    *inputs:*
        I: 3D image

        sigmas: list of sigmas for gaussian filtering before computing derivatives

        justfeatnames: True or False; if to output just feature names
        (this is useful to allocate memory in certain machine learning tasks)

    *output:*
        stack of derivatives if justfeatnames = False, otherwise just feature names;
        for each sigma, the following derivatives are computed:
        d0, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz,
        normGrad (gradient magnitude), normHessDiag (norm of diagonal of hessian)

        derivatives are indexed by the last dimension
    """

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

def imfeatures(I=[],sigmaDeriv=1,sigmaLoG=1,locStatsRad=0,justfeatnames=False):
    """
    computes 2D image features based on *imderivatives*, *imlogfilt*, and *localstats*

    *inputs:*
        I: 2D image, or empty list [] (which is convenient when justfeatnames=True)

        sigmaDeriv: list of sigmas to pass on to *imderivatives*

        sigmaLoG: list of sigmas to use in *imlogfilt*

        locStatsRad: list of radii to use in *localstats*

        justfeatnames: True or False; if to output just feature names
        (this is useful to allocate memory in certain machine learning tasks)

    *output:*
        stack of features, indexed by last dimension
        if justfeatnames = False; otherwise just feature names
    """

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
    """
    computes 3D image features based on *imderivatives3*, *imlogfilt*, and *localstats3*

    *inputs:*
        I: 3D image, or empty list [] (which is convenient when justfeatnames=True)

        sigmaDeriv: list of sigmas to pass on to *imderivatives3*

        sigmaLoG: list of sigmas to use in *imlogfilt*

        sigmaSurf: list of sigmas to use in *imridgelikl*

        locStatsRad: list of radii to use in *localstats3*

        justfeatnames: True or False; if to output just feature names
        (this is useful to allocate memory in certain machine learning tasks)

    *output:*
        stack of features, indexed by last dimension
        if justfeatnames = False; otherwise just feature names
    """

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
    """
    turns a stack of planes into a list of planes

    *input:*
        S: stack of planes; assumes they are indexed by last dimension

    *output:*
        list of planes
    """

    L = []
    for i in range(size(S)[2]):
        L.append(S[:,:,i])
    return L

def list2stack(l):
    """
    stacks a list of planes

    *input:*
        l: list of planes

    *output:*
        stack, with planes indexed by the first dimension
    """

    n = len(l)
    nr = l[0].shape[0]
    nc = l[0].shape[1]
    S = np.zeros((n,nr,nc)).astype(l[0].dtype)
    for i in range(len(l)):
        S[i,:,:] = l[i]
    return S

def thrsegment(I,wsBlr,wsThr): # basic threshold segmentation
    """
    basic threshold segmentation

    *inputs:*
        I: image

        wsBlr: blur parameter, between 0 and 1; 0 implies a gaussian filtering
        with sigma 1, 1 implies a gaussian filtering with sigma 5

        wsThr: threshold parameter, between 0 and 1

    *output:*
        binary threholded image
    """

    G = imgaussfilt(I,sigma=(1-wsBlr)+wsBlr*5) # min 1, max 5
    M = G > wsThr
    return M

def circleKernel(radius,sigma,ftype):
    """
    circular kernel for circular shape detection

    *inputs:*
        radius: radius of circle

        sigma: sigma of gaussian or LoG kernel (see ftype parameter)

        ftype: 'log' or 'gauss'; use 'log' for kernel appropriate for 'empty circle' detection;
        use 'gauss' for kernel apropriate for 'full circle' detection

    *output:*
        kernel
    """

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
    """
    morlet wavelet kernel

    *inputs:*
        stretch: stretch (elongation) parameter

        scale: scale parameter

        orientation: orientation parameter (in degrees)

    *outputs:*
        mr, mi: the real and imaginary parts of the kernel, respectively
    """

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
    """
    2D convolution

    *inputs:*
        I: 2D image

        K: 2D kernel

        m: mode ('full', 'valid', 'same')

    *output:*
        convolved image
    """

    return convolve(I, K, mode=m)

def conv3(I,K,m):
    """
    3D convolution

    *inputs:*
        I: 3D image

        K: 3D kernel

        m: mode ('full', 'valid', 'same')

    *output:*
        convolved image
    """

    return convolve(I, K, mode=m)

def centerCrop(I,nr,nc):
    """
    crops center portion of 2D, single channel image

    *inputs:*
        I: image

        nr: number of rows of output crop

        nr: number of columns of output crop

    *output:*
        crop
    """

    nrI = I.shape[0]
    ncI = I.shape[1]
    r0 = int(nrI/2)
    c0 = int(ncI/2)
    nr2 = int(nr/2)
    nc2 = int(nc/2)
    return I[r0-nr2:r0+nr2,c0-nc2:c0+nc2]

def centerCropMultChan(I,nr,nc):
    """
    crops center portion of 2D, multi channel image; assumes channels are on 1sd dimension

    *inputs:*
        I: image

        nr: number of rows of output crop

        nr: number of columns of output crop

    *output:*
        crop
    """

    nrI = I.shape[1]
    ncI = I.shape[2]
    r0 = int(nrI/2)
    c0 = int(ncI/2)
    nr2 = int(nr/2)
    nc2 = int(nc/2)
    return I[:,r0-nr2:r0+nr2,c0-nc2:c0+nc2]

def pad(I,k):
    """
    pads image with zeros

    *inputs:*
        I: 2D, single channel image

        k: ammount to pad at each border

    *output:*
        padded image
    """

    J = np.zeros((I.shape[0]+2*k,I.shape[1]+2*k))
    J[k:-k,k:-k] = I
    return J

def fullPatchCoordinates2D(nr,nc,patchSize):
    """
    patch coordinates of a 2D image split without overlap

    *inputs:*
        nr: number of image rows

        nc: number of image columns

        patchSize: size of each side of a square patch

    *output:*
        list of patch coordinates [[row0, row1, col0, col1], ...]

    *note:*
        assumes nr and nc are multiples of patchSize
    """

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
    """
    patch coordinates of a 3D image split without overlap

    *inputs:*
        nz: number of image planes

        nr: number of image rows

        nc: number of image columns

        patchSize: size of each side of a cubic patch

    *output:*
        list of patch coordinates [[pln0, pln1, row0, row1, col0, col1], ...]

    *note:*
        assumes nz, nr, nc are multiples of patchSize
    """

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
    """
    builds mosaic from a stack of 2D images

    *input:*
        S: stack; assumes planes are indexed by first coordinate

    *output:*
        mosaic, of type uint8, regardless of input type
    """

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
    """
    changes view plane of 3D image, via numpy's *moveaxis* function

    *inputs:*
        I: 3D image

        currentViewPlane: 'x', 'y' or 'z'; current view plane

        newViewPlane: 'x', 'y', or 'z'; output view plane

    *output:*
        3D image with moved view plane
    """

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
    """
    laplacian of gaussian 3D kernel

    *input:*
        sigma: kernel sigma parameter

    *output:*
        3D kernel
    """

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
    """
    kernels for 3D surface detection

    *inputs:*
        sigma: sigma of kernels

        radius: controls size of kernel, which is 2*(radius + 2*sigma)+1; 
        if radius = None, it's replaced with radius = 4*sigma

        quantity: number of surface kernels; options are 3, 4, 6, 10

    *output:*
        list of surface kernels
    """

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
    """
    highlights ridges in 2D or 3D image

    *inputs:*
        I: image, 2D or 3D

        sigma: scale of ridges

    *output:*
        ridge likelihood image
    """

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
    """
    finds spots in 2D image

    *inputs:*
        I: 2D image; should be double, in range [0, 1]

        sigma: scale of spots; should be an integer: 1, 2, 3, ...

        thr: correlation threshold for spot selection; should be in range (0, 1), where 1
        means identically correlated to 'ideal' spot

    *outputs:*
        psList: list of candidate spots

        sPsList: list of selected spots

        I3: RGB image visualizing candidate and selected spots
    """

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
    """
    finds spots in 3D image

    *inputs:*
        I: 3D image; should be double, in range [0, 1]

        sigma: scale of spots; should be an integer: 1, 2, 3, ...

        thr: correlation threshold for spot selection; should be in range (0, 1), where 1
        means identically correlated to 'ideal' spot

    *outputs:*
        psList: list of candidate spots

        sPsList: list of selected spots

        S: 3D image visualizing selected spots
    """

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
    """
    makes an RGB image out of 2D planes

    *inputs*:
        R: 'red' image plane

        G: 'green' image plane

        B: 'blue' image plane

    *ouput:*
        RGB image
    """

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
    """
    applies otsu threshold

    *input:*
        I: image

    *output:*
        binary thresholded image
    """

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
    """
    interpolates plane annotations on 3D images

    *inputs:*
        A: annotation image

        classIdx: index of class to interpolate

    *output:*
        image with interpolated anotations\

    *note:*
        this function uses bwInterpSingleObjectMasks to 'fill in' planes in between
        those which have been annotated with label *classIdx*; it assumes
        annotations on each plane correspond to a single closed contour
    """

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

def splitIntoTiles2D(pathIn, pathOut):
    """
    splits images/annotations into tiles for unet2D model training

    *inputs:*
        pathIn: input folder path; expects folder containing a list of pairs (image, annotation)
        where each pair is named following the pattern 'rootname_Img.tif', 'rootname_Ant.tif';
        _Img.tif is expected to be a single-channel 2D image; _Ant.tif is expected to be
        uint8, where pixels labeled 1, 2, 3, etc correspond to class 1, 2, 3, etc, respectively

        pathOut: output folder path; the function will get patches of size 60x60 every 20 pixels
        in each dimension, re-index them, and write to the output folder; patches where there
        are no labeled pixels (all pixels from _Ant.tif in that patch are 0) will be discarded
    """

    l_I = listfiles(pathIn, '_Img.tif')
    l_A = listfiles(pathIn, '_Ant.tif')

    count = -1
    for index in range(len(l_I)):
        I = tifread(l_I[index])
        A = tifread(l_A[index])

        for i in range(0, A.shape[0]-60, 20):
            for j in range(0, A.shape[1]-60, 20):
                block_A = np.copy(A[i:i+60, j:j+60])
                block_A_crop = np.copy(block_A[20:40, 20:40])
                mask_block_A_crop = block_A_crop > 0
                if np.any(mask_block_A_crop):
                    count += 1
                    print('index', index, '| block:', count, '| # labels:', np.sum(mask_block_A_crop))

                    block_A[:] = 0
                    block_A[20:40, 20:40] = block_A_crop
                    tifwrite(block_A, pathjoin(pathOut, 'I%05d_Ant.tif' % count))

                    block_I = np.copy(I[i:i+60, j:j+60])
                    tifwrite(block_I, pathjoin(pathOut, 'I%05d_Img.tif' % count))

def splitIntoTiles3D(pathIn, pathOut):
    """
    splits images/annotations into tiles for unet3D model training

    *inputs:*
        pathIn: input folder path; expects folder containing a list of pairs (image, annotation)
        where each pair is named following the pattern 'rootname_Img.tif', 'rootname_Ant.tif';
        _Img.tif is expected to be a single-channel 2D image; _Ant.tif is expected to be
        uint8, where pixels labeled 1, 2, 3, etc correspond to class 1, 2, 3, etc, respectively

        pathOut: output folder path; the function will get patches of size 60x60 every 20 pixels
        in each dimension, re-index them, and write to the output folder; patches where there
        are no labeled pixels (all pixels from _Ant.tif in that patch are 0) will be discarded
    """

    l_I = listfiles(pathIn, '_Img.tif')
    l_A = listfiles(pathIn, '_Ant.tif')

    count = -1
    for index in range(len(l_I)):
        I = tifread(l_I[index])
        A = tifread(l_A[index])

        for i in range(0, A.shape[0]-60, 20):
            for j in range(0, A.shape[1]-60, 20):
                for k in range(0, A.shape[2]-60, 20):
                    block_A = np.copy(A[i:i+60, j:j+60, k:k+60])
                    block_A_crop = np.copy(block_A[20:40, 20:40, 20:40])
                    mask_block_A_crop = block_A_crop > 0
                    if np.any(mask_block_A_crop):
                        count += 1
                        print('index', index, '| block:', count, '| # labels:', np.sum(mask_block_A_crop))

                        block_A[:] = 0
                        block_A[20:40, 20:40, 20:40] = block_A_crop
                        tifwrite(block_A, pathjoin(pathOut, 'I%05d_Ant.tif' % count))

                        block_I = np.copy(I[i:i+60, j:j+60, k:k+60])
                        tifwrite(block_I, pathjoin(pathOut, 'I%05d_Img.tif' % count))


def boxes_IoU(box_a, box_b):
    """
    computes the intersection over union coefficient for boxes

    *inputs:*
        box_a, box_b: box coordinate lists [xmin, ymin, xmax, ymax]

    *output:*
        area of intersection divided by area of union
    """

    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    xmin_b, ymin_b, xmax_b, ymax_b = box_b

    min_ymax = np.minimum(ymax_a, ymax_b)
    max_ymin = np.maximum(ymin_a, ymin_b)
    min_xmax = np.minimum(xmax_a, xmax_b)
    max_xmin = np.maximum(xmin_a, xmin_b)

    x_intersection = np.maximum(min_xmax-max_xmin, 0)
    y_intersection = np.maximum(min_ymax-max_ymin, 0)
    
    area_intersection = x_intersection*y_intersection
    area_a = (xmax_a-xmin_a)*(ymax_a-ymin_a)
    area_b = (xmax_b-xmin_b)*(ymax_b-ymin_b)
    area_union = area_a+area_b-area_intersection

    if area_union == 0:
        return 0.0

    return area_intersection/area_union

def boxes_intersect(box_a, box_b):
    """
    checks if two boxes intersect

    *inputs:*
        box_a, box_b: box coordinate lists [xmin, ymin, xmax, ymax]

    *output:*
        True if boxes intersect, otherwise False
    """

    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    xmin_b, ymin_b, xmax_b, ymax_b = box_b

    min_ymax = np.minimum(ymax_a, ymax_b)
    max_ymin = np.maximum(ymin_a, ymin_b)
    min_xmax = np.minimum(xmax_a, xmax_b)
    max_xmin = np.maximum(xmin_a, xmin_b)

    x_intersection = np.maximum(min_xmax-max_xmin, 0)
    y_intersection = np.maximum(min_ymax-max_ymin, 0)
    
    return x_intersection*y_intersection > 0

def masks_IoU(mask_a, mask_b):
    """
    computes the intersection over union coefficient for masks

    *inputs:*
        mask_a, mask_b: masks (binary images of the same size)

    *output:*
        area of intersection divided by area of union
    """

    area_intersection = np.sum(mask_a*mask_b)
    area_union = np.sum(mask_a)+np.sum(mask_b)-area_intersection
    
    if area_union == 0:
        return 0.0

    return area_intersection/area_union

def labels_to_boxes_and_contours(label_image):
    """
    finds object bounding boxes and contours from label image

    *input:*
        label image

    *outputs:*
        boxes: list of boxes where each box is a list [xmin, ymin, xmax, ymax]

        contours: (n,2) array of (row, col) locations of contour points
    """

    obj_ids = np.unique(label_image)
    obj_ids = obj_ids[1:]

    masks = label_image == obj_ids[:, None, None]

    num_objs = len(obj_ids)
    boxes = []
    contours = []
    for i in range(num_objs):
        mi = masks[i]
        pos = np.where(mi)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

        F = np.zeros(mi.shape, dtype=bool)
        F[0,:] = True; F[-1,:] = True; F[:,0] = True; F[:,-1] = True
        F = np.logical_and(F, mi)
        CT = np.logical_and(mi, np.logical_not(imerode(mi,1)))
        CT = np.logical_or(CT, F)
        ct = np.argwhere(CT)

        contours.append(ct)
        
    return boxes, contours

def mask2label(mask):
    """
    labels individual objects in binary mask

    *input:*
        mask: binary mask

    *output:*
        image where each object has pixels of identical, unique value
    """

    return label(mask)

def generateSynthCellsImage(im_size=800, n_circs=50, rad_min=30, rad_max=40, dist_factor=0.99):
    """
    generates synthetic image with cells (circles), and corresponding label image

    *inputs:*
        im_size: output image will be square of size im_size by im_size

        n_circs: number of circles (cells)

        rad_min: minimum cell radius

        rad_max: maximum cell radius

        dist_factor: float > 0; if dist_factor < 1, cells may overlap;
        if dist_factor > 1, cells will not overlap; note that if dist_factor is too large,
        the function may enter an infinite loop
        since it may not be possible to draw n_circs with enough separation;
        in general, all parameters should be picked so that drawing n_circs in an image
        of size im_size by im_size is feasible

    *outputs:*
        I: double, range [0,1], grayscale image containing n_circs 

        L: corresponding label image
    """

    itv = np.arange(im_size)
    X, Y = np.meshgrid(itv, itv)
    I = 0.8*np.random.rand()+0.2*np.random.rand(im_size,im_size)
    L = np.zeros(I.shape, dtype=np.uint8)

    xyr = []
    i_circ = 0
    while i_circ < n_circs:
        if xyr:
            CS = np.array(xyr)
            while True:
                r0 = np.random.randint(im_size)
                c0 = np.random.randint(im_size)
                rd = np.random.randint(rad_min, rad_max)
                C0 = repmat([r0, c0, rd], len(xyr), 1)
                ds = C0[:,:2]-CS[:,:2]
                ds = np.sqrt(np.sum(ds**2, axis=1))
                if not np.any(ds < dist_factor*(C0[:,2]+CS[:,2])):
                    xyr.append([r0, c0, rd])
                    M = np.sqrt((X-r0)**2+(Y-c0)**2) < rd
                    R = 0.2*np.random.rand(im_size, im_size)
                    I[M] = 0.8*np.random.rand()+R[M]
                    i_circ += 1
                    L[M] = i_circ
                    break
        else:
            r0 = np.random.randint(im_size)
            c0 = np.random.randint(im_size)
            rd = np.random.randint(rad_min, rad_max)
            xyr.append([r0, c0, rd])
            M = np.sqrt((X-r0)**2+(Y-c0)**2) < rd
            R = 0.2*np.random.rand(im_size, im_size)
            I[M] = 0.8*np.random.rand()+R[M]
            i_circ += 1
            L[M] = i_circ

    return I, L

def generateSynthCellsImage3D(im_size=100, n_spheres=10, rad_min=10, rad_max=20, dist_factor=0.99):
    """
    generates synthetic 3D image with cells (spheres), and corresponding label image

    *inputs:*
        im_size: output image will be a cube of size im_size^3

        n_spheres: number of spheres (cells)

        rad_min: minimum cell radius

        rad_max: maximum cell radius

        dist_factor: float > 0; if dist_factor < 1, cells may overlap;
        if dist_factor > 1, cells will not overlap; note that if dist_factor is too large,
        the function may enter an infinite loop
        since it may not be possible to draw n_spheres with enough separation;
        in general, all parameters should be picked so that drawing n_spheres in an image
        of size im_size^3 is feasible

    *outputs:*
        I: double, range [0,1], grayscale 3D image containing n_spheres

        L: corresponding label image
    """

    itv = np.arange(im_size)
    X, Y, Z = np.meshgrid(itv, itv, itv)
    I = 0.8*np.random.rand()+0.2*np.random.rand(im_size,im_size,im_size)
    L = np.zeros(I.shape, dtype=np.uint8)

    xyzr = []
    i_sphere = 0
    while i_sphere < n_spheres:
        if xyzr:
            CS = np.array(xyzr)
            while True:
                r0 = np.random.randint(im_size)
                c0 = np.random.randint(im_size)
                p0 = np.random.randint(im_size)
                rd = np.random.randint(rad_min, rad_max)
                C0 = repmat([r0, c0, p0, rd], len(xyzr), 1)
                ds = C0[:,:3]-CS[:,:3]
                ds = np.sqrt(np.sum(ds**2, axis=1))
                if not np.any(ds < dist_factor*(C0[:,3]+CS[:,3])):
                    xyzr.append([r0, c0, p0, rd])
                    M = np.sqrt((X-r0)**2+(Y-c0)**2+(Z-p0)**2) < rd
                    R = 0.2*np.random.rand(im_size, im_size, im_size)
                    I[M] = 0.8*np.random.rand()+R[M]
                    i_sphere += 1
                    L[M] = i_sphere
                    break
        else:
            r0 = np.random.randint(im_size)
            c0 = np.random.randint(im_size)
            p0 = np.random.randint(im_size)
            rd = np.random.randint(rad_min, rad_max)
            xyzr.append([r0, c0, p0, rd])
            M = np.sqrt((X-r0)**2+(Y-c0)**2+(Z-p0)**2) < rd
            R = 0.2*np.random.rand(im_size, im_size, im_size)
            I[M] = 0.8*np.random.rand()+R[M]
            i_sphere += 1
            L[M] = i_sphere

    return I, L
