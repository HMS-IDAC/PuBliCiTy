import numpy as np
import pixelclassifier as pc
import voxelclassifier as vc
from gpfunctions import *
import pickle


# -------------------------

class ImageProcessor:
    def __init__(self,**kwargs):
        self.args = kwargs

class IPBlur(ImageProcessor):
    def run(self,I):
        return imgaussfilt(I,self.args['sigma'])

class IPLoG(ImageProcessor):
    def run(self,I):
        return imlogfilt(I,self.args['sigma'])

class IPGradMag(ImageProcessor):
    def run(self,I):
        return imgradmag(I,self.args['sigma'])

class IPDerivatives(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 2:
            return np.moveaxis(imderivatives(I,self.args['sigma']),[2,0,1],[0,1,2])
        if len(I.shape) == 3:
            return np.moveaxis(imderivatives3(I,self.args['sigma']),[0,3,1,2],[0,1,2,3])
        if len(I.shape) > 3:
            print('case len(I.shape) > 3 not considered')

class IPThresholdSegment(ImageProcessor):
    def run(self,I):
        C = I
        if self.args['pcIdx'] == 0: # dark pixels
            C = 1-C
        M = thrsegment(C,self.args['segBlr'],self.args['segThr'])
        if self.args['returnType'] == 'int':
            return M.astype(int)
        if self.args['returnType'] == 'double':
            return M.astype('float64')

class IPPixelClassifierSegment(ImageProcessor):
    def run(self,I):
        C = pc.classify(I,self.args['model'],output='probmaps')
        M = thrsegment(C[:,:,self.args['pmIdx']],self.args['segBlr'],self.args['segThr'])
        if self.args['returnType'] == 'int':
            return M.astype(int)
        if self.args['returnType'] == 'double':
            return M.astype('float64')

class IPVoxelClassifierSegment(ImageProcessor):
    def run(self,I):
        C = vc.classify(I,self.args['model'],output='probmaps')
        M = thrsegment(C[:,:,:,self.args['pmIdx']],self.args['segBlr'],self.args['segThr'])
        if self.args['returnType'] == 'int':
            return M.astype(int)
        if self.args['returnType'] == 'double':
            return M.astype('float64')

class IPMachLearnProbMaps(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 2:
            return np.moveaxis(pc.classify(I,self.args['model'],output='probmaps'),[2,0,1],[0,1,2])
        if len(I.shape) == 3:
            return np.moveaxis(vc.classify(I,self.args['model'],output='probmaps'),[0,3,1,2],[0,1,2,3])
        if len(I.shape) > 3:
            print('case len(I.shape) > 3 not considered')

class IPExtractPlane(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 3:
            return I[self.args['index'],:,:]
        else:
            print('invalid tensor dimension')

class IPExtractChannel(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 4:
            return I[:,self.args['index'],:,:]
        else:
            print('invalid tensor dimension')

class IPMedianFilter(ImageProcessor):
    def run(self,I):
        # if len(I.shape) == 3:
        #     patchSize = 50
        #     margin = 10
        #     mode = 'accumulate'
        #     PI3D.setup(I,patchSize,margin,mode)
        #     PI3D.createOutput()
        #     for i in range(PI3D.NumPatches):
        #         print(i)
        #         P = PI3D.getPatch(i)
        #         PI3D.patchOutput(i,medfilt(P,self.args['size']))
        #     J = PI3D.getValidOutput()
        #     print(np.max(I),np.min(I),np.max(J),np.min(J))
        #     return J

        return medfilt(I,self.args['size'])

class IPMaximumFilter(ImageProcessor):
    def run(self,I):
        return maxfilt(I,self.args['size'])

class IPMinimumFilter(ImageProcessor):
    def run(self,I):
        return minfilt(I,self.args['size'])

class IPViewPlane(ImageProcessor):
    def run(self, I):
        if len(I.shape) == 3:
            if self.args['currentViewPlane'] == 'z':
                if self.args['plane'] == 'x':
                    return np.moveaxis(I,[2,0,1],[0,1,2])
                if self.args['plane'] == 'y':
                    return np.moveaxis(I,[1,2,0],[0,1,2])
                else:
                    return I
            if self.args['currentViewPlane'] == 'y':
                if self.args['plane'] == 'z':
                    return np.moveaxis(I,[2,0,1],[0,1,2])
                if self.args['plane'] == 'x':
                    return np.moveaxis(I,[1,2,0],[0,1,2])
                else:
                    return I
            if self.args['currentViewPlane'] == 'x':
                if self.args['plane'] == 'y':
                    return np.moveaxis(I,[2,0,1],[0,1,2])
                if self.args['plane'] == 'z':
                    return np.moveaxis(I,[1,2,0],[0,1,2])
                else:
                    return I
        else:
            print('IPViewPlane only applicable to 3D images')



# -------------------------

class DL: # dialog logic
    def __init__(self):
        self.newImageFromServerStep = 0
        self.saveCurrentImageToServerStep = 0
        self.saveAnnotationsToServerStep = 0
        self.loadAnnotationsFromServerStep = 0
        self.editAnnotationsStep = 0
        self.saveMLModelToServerStep = 0
        self.loadMLModelFromServerStep = 0
        self.ipOperation = None
        self.ipParameters = None
        self.ipStep = 0
        self.loginStep = 0

# -------------------------

class IA: # image analysis
    def __init__(self):
        self.I = None # current image, fetched by client along with labelmask or segmMask if present
        self.labelMask = None
        self.interpolationMask = None
        self.labelMaskIndex = None
        self.segmMask = None
        self.maskType = None  # 'noMask', 'labelMask', 'segmMask'
        self.imagesOnServer = None # list of paths to images on server
        self.unsavedAnnotationsOnServer = None # list of paths to annotations on server not yet saved (to subfolders)
        self.mlModelsOnServer = None # list of paths to ML models on server
        self.annotationSetsOnServer = None # list of folders containing annotations
        self.hasImage = False
        self.imageType = None
        self.imageShape = None
        self.planeIndex = None
        self.channelIndex = None
        self.nClasses = None
        self.mlSegmenterTrainParameters = None
        self.mlSegmenterModel = None
        self.mlSegmenterAnnotationsPath = None
        self.originalImage = None
        self.nPlanes = 0
        self.nChannels = 0
        self.currentViewPlane = None
        self.username = None
        self.userFolder = None
        self.dataSubfolder = None
        self.scratchSubfolder = None
        self.annotationsSubfolder = None
        self.modelsSubfolder = None
        self.onMobile = None
        self.coords = None # row, col coordinates (e.g. locations of spots)


    def setupDirectories(self,mainFolder):
        # for now just setup paths; directories should be created manually
        self.userFolder = pathjoin(mainFolder,self.username)
        self.dataSubfolder = pathjoin(self.userFolder,'Data')
        self.scratchSubfolder = pathjoin(self.userFolder,'Scratch')
        self.annotationsSubfolder = pathjoin(pathjoin(self.userFolder,'MachineLearning'),'Annotations')
        self.modelsSubfolder = pathjoin(pathjoin(self.userFolder,'MachineLearning'),'Models')
        self.cleanDirectories()

    def cleanDirectories(self):
        l = listfiles(self.annotationsSubfolder,'.tif')
        for i in range(len(l)):
            removeFile(l[i])
        for fType in ['.tif','.zip','.csv']:
            l = listfiles(self.scratchSubfolder,fType)
            for i in range(len(l)):
                removeFile(l[i])
        removeFolderIfExistent(pathjoin(self.scratchSubfolder,'Annotations'))

    def setImage(self,Image):
        # currentPlaneView, originalImage, maskType set from outside
        # this method can be called to update an existing image (e.g. via image processing)
        self.I = im2double(Image)
        self.hasImage = True
        self.imageShape = Image.shape
        self.planeIndex = None
        self.channelIndex = None
        self.nPlanes = 0
        self.nChannels = 0
        if len(self.imageShape) > 2:
            self.planeIndex = int(self.imageShape[0]/2)
            self.nPlanes = self.imageShape[0]
        if len(self.imageShape) > 3:
            self.channelIndex = 0
            self.nChannels = self.imageShape[1]

    def createLabelMask(self, prepareForInterpolation):
        self.labelMaskIndex = None
        if len(self.imageShape) < 3:
            self.labelMask = np.uint8(np.zeros(self.imageShape))
        elif len(self.imageShape) == 3:
            self.labelMask = np.uint8(np.zeros(self.imageShape))
            if prepareForInterpolation:
                self.interpolationMask = np.copy(self.labelMask)
            else:
                self.interpolationMask = None
        elif len(self.imageShape) == 4: # 3D annotation volume
            print('4D image case not considered yet')
            # self.labelMask = np.uint8(np.zeros((self.imageShape[0],self.imageShape[2],self.imageShape[3])))
        else:
            print('5D image case not considered')

    def updateLabelMask(self,ant,ant4itp):
        if len(self.imageShape) == 2:
            self.labelMask[:] = 0
            for i in range(len(ant)):
                indices = ant[i]
                for j in range(len(indices)):
                    row = int(indices[j]/self.imageShape[1])
                    col = int(indices[j]-row*self.imageShape[1])
                    self.labelMask[row,col] = i+1
        elif len(self.imageShape) == 3:# or len(self.imageShape) == 4:
            self.labelMask[self.planeIndex,:,:] = 0
            for i in range(len(ant)):
                indices = ant[i]
                for j in range(len(indices)):
                    row = int(indices[j]/self.imageShape[-1])
                    col = int(indices[j]-row*self.imageShape[-1])
                    self.labelMask[self.planeIndex,row,col] = i+1
            if self.interpolationMask is not None:
                for i in range(len(ant4itp)):
                    indices = ant4itp[i]
                    for j in range(len(indices)):
                        row = int(indices[j]/self.imageShape[-1])
                        col = int(indices[j]-row*self.imageShape[-1])
                        self.interpolationMask[self.planeIndex,row,col] = i+1
                print('-'*10)

    def thresholdSegment(self,pcIdx,segBlr,segThr): # works for 2D and 3D images
        ip = IPThresholdSegment(pcIdx=pcIdx,segBlr=segBlr,segThr=segThr,returnType='int')
        self.segmMask = ip.run(self.I)

    def findSpots(self,sigma,threshold): # for now only works in 2D
        _, sPsList, _ = findSpots2D(self.I,sigma,thr=threshold)
        self.coords = sPsList

    def prepTableFromCoords(self): # for now assumes coords are 2D spot locations
        import csv
        with open(pathjoin(self.scratchSubfolder,'Scratch.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['row','col'])
            [writer.writerow(r) for r in self.coords]

    def interpolatePlaneMasks(self):
        t0 = tic()
        for classIdx in range(1,self.nClasses+1):
            itpClassIdx = interpolateAnnotations3D(self.interpolationMask, classIdx)
            self.labelMask[itpClassIdx == classIdx] = itpClassIdx[itpClassIdx == classIdx]
        t1 = tic()
        print('-'*10)
        print('time spent interpolating:', t1-t0)
        print('-'*10)
        if t1-t0 < 1:
            pause(1-(t1-t0)) # to avoid annotation tool re-display error
        self.interpolationMask[:] = 0

    def saveLabelMask(self,path):
        if len(self.labelMask.shape) == 2:
            I2Save = self.I
            L2Save = self.labelMask
        elif len(self.labelMask.shape) == 3:
            ip = IPViewPlane(plane='z',currentViewPlane=self.currentViewPlane)
            I2Save = ip.run(self.I)
            L2Save = ip.run(self.labelMask)
        else:
            print('saveLabelMask case dimension > 3 not considered')

        if self.labelMaskIndex is None:
            idx = len(listfiles(path,'_Img.tif'))
        else:
            idx = self.labelMaskIndex

        tifwrite(L2Save, pathjoin(path, 'Image_%03d_%03d_Ant.tif' % (self.nClasses,idx)))
        tifwrite(np.uint16(65535*I2Save), pathjoin(path, 'Image_%03d_%03d_Img.tif' % (self.nClasses,idx)))

    def trainMLSegmenter(self):
        sigmaDeriv = self.mlSegmenterTrainParameters[0]
        sigmaLoG = self.mlSegmenterTrainParameters[1]

        imSh = tifread(listfiles(self.mlSegmenterAnnotationsPath,'_Img.tif')[0]).shape

        if len(imSh) == 2:
            print('training pixel classifier')
            self.mlSegmenterModel = pc.train(self.mlSegmenterAnnotationsPath,sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,locStatsRad=0)
            # pc.plotFeatImport(self.mlSegmenterModel['featImport'],self.mlSegmenterModel['featNames'])
        elif len(imSh) == 3:
            print('training voxel classifier')
            self.mlSegmenterModel = vc.train(self.mlSegmenterAnnotationsPath,sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,locStatsRad=0)
            # pc.plotFeatImport(self.mlSegmenterModel['featImport'],self.mlSegmenterModel['featNames'])
        # l = listfiles(self.mlSegmenterAnnotationsPath, '.tif')
        # [os.remove(l[i]) for i in range(len(l))]

    def mlSegment(self,pmIdx,segBlr,segThr):
        if len(self.imageShape) == 2:
            ip = IPPixelClassifierSegment(model=self.mlSegmenterModel,pmIdx=pmIdx,segBlr=segBlr,segThr=segThr,returnType='int')
        elif len(self.imageShape) == 3:
            ip = IPVoxelClassifierSegment(model=self.mlSegmenterModel,pmIdx=pmIdx,segBlr=segBlr,segThr=segThr,returnType='int')
        self.segmMask = ip.run(self.I)

    def saveAnnotations(self,path):
        print('saving annotations')
        createFolderIfNonExistent(path)
        for i in range(len(self.unsavedAnnotationsOnServer)):
            moveFile(self.unsavedAnnotationsOnServer[i],path)
    def loadAnnotations(self,path):
        print('load annotations from', path)
        [p,n,e] = fileparts(path)
        l = listfiles(p,'.tif')
        for i in range(len(l)):
            removeFile(l[i])
        l = listfiles(path,'.tif')
        for i in range(len(l)):
            copyFile(l[i],p)

    def saveMLModel(self,path):
        print('writing ml model')
        modelFile = open(path, 'wb')
        pickle.dump(self.mlSegmenterModel, modelFile)

    def loadMLModel(self,path):
        print('loading ml model')
        modelFile = open(path, 'rb')
        self.mlSegmenterModel = pickle.load(modelFile)

    def imageProcessing(self,ipOperation,ipParameters):
        if ipOperation == 'blur':
            ip = IPBlur(sigma=ipParameters['sigma'])
        elif ipOperation == 'log':
            ip = IPLoG(sigma=ipParameters['sigma'])
        elif ipOperation == 'gradient magnitude':
            ip = IPGradMag(sigma=ipParameters['sigma'])
        elif ipOperation == 'derivatives':
            ip = IPDerivatives(sigma=ipParameters['sigma'])
        elif ipOperation == 'ml probability maps':
            ip = IPMachLearnProbMaps(model=self.mlSegmenterModel)
        elif ipOperation == 'extract plane':
            ip = IPExtractPlane(index=ipParameters['plane']-1)
        elif ipOperation == 'extract channel':
            ip = IPExtractChannel(index=ipParameters['channel']-1)
        elif ipOperation == 'median filter':
            ip = IPMedianFilter(size=ipParameters['size'])
        elif ipOperation == 'maximum filter':
            ip = IPMaximumFilter(size=ipParameters['size'])
        elif ipOperation == 'minimum filter':
            ip = IPMinimumFilter(size=ipParameters['size'])
        elif ipOperation == 'threshold':
            ip = IPThresholdSegment(pcIdx=ipParameters['pcIdx'],segBlr=ipParameters['segBlr'],segThr=ipParameters['segThr'],returnType='double')
        elif ipOperation == 'mlthreshold':
            if len(self.imageShape) == 2:
                ip = IPPixelClassifierSegment(model=self.mlSegmenterModel,pmIdx=ipParameters['pmIdx'],segBlr=ipParameters['segBlr'],segThr=ipParameters['segThr'],returnType='double')
            elif len(self.imageShape) == 3:
                ip = IPVoxelClassifierSegment(model=self.mlSegmenterModel,pmIdx=ipParameters['pmIdx'],segBlr=ipParameters['segBlr'],segThr=ipParameters['segThr'],returnType='double')
            self.segmMask = ip.run(self.I)
        elif ipOperation == 'view plane':
            ip = IPViewPlane(plane=ipParameters['plane'],currentViewPlane=ipParameters['currentViewPlane'])
        self.setImage(ip.run(self.I))
        if len(self.imageShape) > 2:
            self.currentViewPlane = 'z'
        
        if ipOperation == 'view plane':
            if self.maskType == 'segmMask':
                self.segmMask = ip.run(self.segmMask)
            elif self.maskType == 'labelMask':
                self.labelMask = ip.run(self.labelMask)
                if self.interpolationMask is not None:
                    self.interpolationMask = ip.run(self.interpolationMask)
