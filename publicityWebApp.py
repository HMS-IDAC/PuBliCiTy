from flask import Flask, request, render_template, jsonify, send_file, make_response
from flask_socketio import SocketIO, emit
# from werkzeug import secure_filename
from werkzeug.utils import secure_filename
import time
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from PuBliCiTyWebApp import IA, DL
import json
import threading
from gpfunctions import *
import shutil
import random
import string


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32) # 'secret!' # http://flask.pocoo.org/docs/1.0/config/
socketio = SocketIO(app, async_mode='threading')
# https://github.com/miguelgrinberg/Flask-SocketIO/blob/master/example/app.py
# https://flask-socketio.readthedocs.io/en/latest/#version-compatibility
# https://cdnjs.com/libraries/socket.io (see line 15 in index.html)

devMode = False # set to True during development to deploy to 'localhost' (but see note at bottom under 'if __name__ == '__main__'')

mainIA = IA()
mainDL = DL()

MAIN_FOLDER = 'Server'
mainIA.username = 'User'

def randString():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(32))

def shape2String(shape):
    s = '('
    for i in range(len(shape)-1):
        s = s+'%d, ' % shape[i]
    return s+'%d)' % shape[-1]

def pathList2String(l):
    plStr = ''
    for i in range(len(l)-1):
        [p,n,e] = fileparts(l[i])
        idx = '(%d) ' % (i+1)
        plStr = plStr+idx+n+e+', '
    [p,n,e] = fileparts(l[-1])
    idx = '(%d) ' % (len(l))
    plStr = plStr+idx+n+e
    return plStr

def jsonifyCurrentPlane():
    shape = mainIA.imageShape
    maskType = mainIA.maskType
    if len(shape) == 2:
        uint8I = np.uint8(255*normalize(mainIA.I))
        if maskType == 'noMask':
            return jsonify(tcpi=[0,0,0],shape=shape,data=uint8I.tolist(),planeView=mainIA.currentViewPlane,mask=[],maskType='noMask')
        if maskType == 'labelMask':
            mask = mainIA.labelMask
        elif maskType == 'segmMask':
            mask = mainIA.segmMask
        return jsonify(tcpi=[0,0,0],shape=shape,data=uint8I.tolist(),planeView=mainIA.currentViewPlane,mask=mask.tolist(),maskType=maskType)            
    if len(shape) == 3: # plane, row, col
        uint8I = np.uint8(255*normalize(mainIA.I[mainIA.planeIndex,:,:]))
        if maskType == 'noMask':
            return jsonify(tcpi=[0,0,mainIA.planeIndex],shape=shape,data=uint8I.tolist(),planeView=mainIA.currentViewPlane,mask=[],maskType='noMask')
        if maskType == 'labelMask':
            mask = mainIA.labelMask[mainIA.planeIndex,:,:]
        elif maskType == 'segmMask':
            mask = mainIA.segmMask[mainIA.planeIndex,:,:]
        return jsonify(tcpi=[0,0,mainIA.planeIndex],shape=shape,data=uint8I.tolist(),planeView=mainIA.currentViewPlane,mask=mask.tolist(),maskType=maskType)
    if len(shape) == 4: # plane, channel, row, col
        uint8I = np.uint8(255*normalize(mainIA.I[mainIA.planeIndex,mainIA.channelIndex,:,:]))
        if maskType == 'noMask':
            return jsonify(tcpi=[0,mainIA.channelIndex,mainIA.planeIndex],shape=shape,data=uint8I.tolist(),planeView=mainIA.currentViewPlane,mask=[],maskType='noMask')
        if maskType == 'labelMask':
            mask = mainIA.labelMask[mainIA.planeIndex,:,:]
        elif maskType == 'segmMask':
            mask = mainIA.segmMask[mainIA.planeIndex,:,:]
        return jsonify(tcpi=[0,mainIA.channelIndex,mainIA.planeIndex],shape=shape,data=uint8I.tolist(),planeView=mainIA.currentViewPlane,mask=mask.tolist(),maskType=maskType)

def ipDialogStep1(ipOp, followUpQuestion):
    if mainIA.hasImage:
        mainDL.ipOperation = ipOp
        mainDL.ipStep = 1
        emit('dialog', followUpQuestion)
        emit('server2ClientMessage', 'animate last dialog message')
    else:
        emit('dialog', 'set image first')

def extractPlaneDialog():
    if mainIA.hasImage and mainIA.nPlanes > 0 and mainIA.nChannels == 0:
        mainDL.ipOperation = 'extract plane'
        mainDL.ipStep = 1
        emit('dialog', 'which plane? [1,...,%d]' % mainIA.nPlanes)
        emit('server2ClientMessage', 'animate last dialog message')
    else:
        emit('dialog', 'operation invalid for this image')

def extractChannelDialog():
    if mainIA.hasImage and mainIA.nChannels > 0:
        mainDL.ipOperation = 'extract channel'
        mainDL.ipStep = 1
        emit('dialog', 'which channel? [1,...,%d]' % mainIA.nChannels)
        emit('server2ClientMessage', 'animate last dialog message')
    else:
        emit('dialog', 'operation invalid for this image')

# 'connect' messages are test messages
# 'server2ClientMessage' messages are hidden to the user
# 'dialog' messages are shown on the dialog area

@socketio.on('connect')                                                         
def connect():             
    print('server side connected')
    emit('server2ClientMessage', 'Server: SocketIO connected')

@socketio.on('client2ServerMessage')
def client2ServerMessage(message):
    if message == 'socket echo test':
        emit('server2ClientMessage', message)
    elif message == 'initialize':
        mainIA.setupDirectories(MAIN_FOLDER)
        emit('server2ClientMessage', 'server initialized')
    elif message[:17] == 'create label mask':
        mainIA.maskType = 'labelMask'
        createLabelMaskOption = int(message[-2])
        prepareForInterpolation = bool(int(message[-1]))
        if createLabelMaskOption == 0: # creating label mask from scratch
            mainIA.createLabelMask(prepareForInterpolation)
        elif createLabelMaskOption == 1: # editing annotation (i.e. label mask already set), thus just tell client to show label mask
            emit('server2ClientMessage', 'fetch plane')
        emit('server2ClientMessage', 'label mask created')
    elif message == 'interpolate plane masks':
        mainIA.interpolatePlaneMasks()
        emit('server2ClientMessage', 'done interpolating plane masks')
        emit('server2ClientMessage', 'fetch plane')
    elif message == 'save label mask':
        mainIA.saveLabelMask(mainIA.annotationsSubfolder)
        mainIA.maskType = 'noMask'
        emit('server2ClientMessage', 'done saving annotations')
    elif message == 'train ml segmenter':
        mainIA.mlSegmenterAnnotationsPath = mainIA.annotationsSubfolder
        mainIA.trainMLSegmenter()
        emit('dialog', 'done training')
    elif message[:10] == 'view plane':
        if mainIA.hasImage and len(mainIA.imageShape) == 3:
            newViewPlane = message[11]
            # print(mainIA.currentViewPlane,'->',newViewPlane)
            mainIA.imageProcessing('view plane',{'plane': newViewPlane, 'currentViewPlane': mainIA.currentViewPlane})
            mainIA.currentViewPlane = newViewPlane
            emit('server2ClientMessage', 'fetch plane')
            # emit('dialog', 'operation %s done' % message)
        else:
            emit('dialog', 'operation available only for 3D images')
    elif message[:13] == 'set mask type':
        mainIA.maskType = message[14:]
        if mainIA.maskType == 'noMask':
            mainIA.labelMask = None
            mainIA.interpolationMask = None
        print('new mask type: ', mainIA.maskType)
    elif message[:19] == 'client is on mobile':
        if message[21] == '0':
            mainIA.onMobile = False
            print('client is on desktop')
        elif message[21] == '1':
            mainIA.onMobile = True
            print('client is on mobile')
    elif message[:9] == 'threshold':
        prmts = []
        splMsg = message.split()
        for i in range(1,len(splMsg)):
            prmts.append(int(splMsg[i]))
        emit('dialog','computing...')
        mainIA.imageProcessing('threshold',{'pcIdx': prmts[0]-1, 'segBlr': prmts[1]/100, 'segThr': prmts[2]/100})
        emit('server2ClientMessage', 'fetch plane')
        emit('dialog','done')
    elif message[:11] == 'mlthreshold':
        prmts = []
        splMsg = message.split()
        for i in range(1,len(splMsg)):
            prmts.append(int(splMsg[i]))
        emit('dialog','computing...')
        mainIA.imageProcessing('mlthreshold',{'pmIdx': prmts[0]-1, 'segBlr': prmts[1]/100, 'segThr': prmts[2]/100})
        emit('server2ClientMessage', 'fetch plane')
        emit('dialog','done')
    elif message[:8] == 'nclasses':
        splMsg = message.split()
        mainIA.nClasses = int(splMsg[1])
        print('nClasses:',mainIA.nClasses)
    elif message[:19] == 'toolFindSpotsButton':
        splMsg = message.split()
        if splMsg[1] == 'done':
            mainIA.prepTableFromCoords()
            emit('dialog','\'find spots\' done | \'download table\' allows downloading a CSV table with spot locations')
        elif splMsg[1] =='cancel':
            mainIA.coords = None
            emit('dialog','\'find spots\' canceled')
    else:
        print('client2ServerMessage', message)

@socketio.on('dialog')
def dialog(message):
    # if message == 'ip address': # hidden (client does not auto-complete)
    #     emit('dialog',mainIA.ipAddress)
    if message == 'image properties':
        if not mainIA.hasImage:
            emit('dialog', 'image not set | enter \'new image\' to set new image')
        else:
            emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(mainIA.imageShape), np.min(mainIA.I), np.max(mainIA.I)))
    elif message == 'new image':
        emit('server2ClientMessage', 'load new image tool')
    elif message == 'new image from server':
        mainDL.newImageFromServerStep = 1
        mainIA.imagesOnServer = listfiles(mainIA.dataSubfolder,'.tif')
        lstr = 'choose image: '+pathList2String(mainIA.imagesOnServer)
        emit('dialog', lstr)
        emit('server2ClientMessage', 'animate last dialog message')
    elif message == 'reset image': # reset to original image (before processing)
        if mainIA.hasImage:
            mainIA.setImage(mainIA.originalImage)
            if len(mainIA.imageShape) == 3:
                mainIA.currentViewPlane = 'z'
            else:
                mainIA.currentViewPlane = ''
            mainIA.maskType = 'noMask'
            mainIA.labelMask = None
            emit('server2ClientMessage', 'fetch plane')
            emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(mainIA.imageShape), np.min(mainIA.I), np.max(mainIA.I)))
        else:
            emit('server2ClientMessage', 'no image to set')
    elif message == 'save current image to server':
        if mainIA.hasImage:
            mainDL.saveCurrentImageToServerStep = 1
            emit('dialog', 'enter image name | first character must be a letter | all characters should be alphanumeric | do not include extension (such as .tif)')
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no image to save')
    elif message == 'download current image':
        if mainIA.onMobile:
            emit('dialog', 'operation only available for desktop clients')
        else:
            if mainIA.hasImage:
                emit('server2ClientMessage', 'fetch image')
            else:
                emit('dialog', 'no image to download')
    elif message == 'save annotations to server':
        mainIA.unsavedAnnotationsOnServer = listfiles(mainIA.annotationsSubfolder,'.tif')
        if mainIA.unsavedAnnotationsOnServer:
            mainDL.saveAnnotationsToServerStep = 1
            emit('dialog', 'enter folder name to save annotations in | first character must be a letter | all characters should be alphanumeric')
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no annotations to save')
    elif message == 'download annotations':
        if mainIA.onMobile:
            emit('dialog', 'operation only available for desktop clients')
        else:
            mainIA.unsavedAnnotationsOnServer = listfiles(mainIA.annotationsSubfolder,'.tif')
            if mainIA.unsavedAnnotationsOnServer:
                emit('server2ClientMessage', 'fetch data')
            else:
                emit('dialog', 'no annotations available | either annotate images or \'load annotations from server\'')
    elif message == 'load annotations from server':
        print('load annotations from server')
        mainIA.annotationSetsOnServer = listsubdirs(mainIA.annotationsSubfolder)
        if mainIA.annotationSetsOnServer:
            mainDL.loadAnnotationsFromServerStep = 1
            lstr = 'choose annotation set: '+pathList2String(mainIA.annotationSetsOnServer)
            emit('dialog', lstr)
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no annotation sets available')
    elif message == 'edit annotations':
        mainIA.unsavedAnnotationsOnServer = listfiles(mainIA.annotationsSubfolder,'.tif')
        if mainIA.unsavedAnnotationsOnServer:
            mainDL.editAnnotationsStep = 1
            emit('dialog', 'enter index of annotations to edit [1,...,%d]' % (len(mainIA.unsavedAnnotationsOnServer)/2))
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no annotations to edit')
    elif message == 'save ml model to server':
        if mainIA.mlSegmenterModel:
            mainDL.saveMLModelToServerStep = 1
            emit('dialog', 'enter model name | first character must be a letter | all characters should be alphanumeric | do not include extension (such as .txt)')
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no model to save')
    elif message == 'load ml model from server':
        mainIA.mlModelsOnServer = listfiles(mainIA.modelsSubfolder,'.ml')
        l = mainIA.mlModelsOnServer
        if len(l) > 0:
            mainDL.loadMLModelFromServerStep = 1
            lstr = 'choose model: '+pathList2String(l)
            emit('dialog', lstr)
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'none available')
    elif message == 'ml probability maps':
        if mainIA.hasImage:
            if mainIA.mlSegmenterModel:
                emit('dialog','computing...')
                mainIA.imageProcessing('ml probability maps',{})
                emit('server2ClientMessage', 'fetch plane')
                emit('dialog', 'operation '+message+' done')
            else:
                emit('dialog', 'no ml model available')
        else:
            emit('dialog', 'set image first')
    elif message == 'find spots':
        emit('server2ClientMessage', 'load find spots tool')
    elif message == 'download table': # e.g. after 'find spots'
        if mainIA.onMobile:
            emit('dialog', 'operation only available for desktop clients')
        else:
            if os.path.isfile(pathjoin(mainIA.scratchSubfolder,'Scratch.csv')):
                emit('server2ClientMessage', 'fetch table')
            else:
                emit('dialog', 'table not available for download')
    elif message == 'median filter':
        ipDialogStep1('median filter', 'which filter radius? [2,3,...]')
    elif message == 'maximum filter':
        ipDialogStep1('maximum filter', 'which filter radius? [2,3,...]')
    elif message == 'minimum filter':
        ipDialogStep1('minimum filter', 'which filter radius? [2,3,...]')
    elif message == 'blur':
        ipDialogStep1('blur', 'which scale? [1,2,3,...]')
    elif message == 'log':
        ipDialogStep1('log', 'which scale? [1,2,3,...]')
    elif message == 'gradient magnitude':
        ipDialogStep1('gradient magnitude', 'which scale? [1,2,3,...]')
    elif message == 'derivatives':
        ipDialogStep1('derivatives', 'which scale? [1,2,3,...]')
    elif message == 'extract plane':
        extractPlaneDialog()
    elif message == 'extract channel':
        extractChannelDialog()
    elif mainDL.newImageFromServerStep == 1:
        if message.isnumeric():
            idx = int(message)-1
            if idx >= 0 and idx < len(mainIA.imagesOnServer):
                mainIA.setImage(tifread(mainIA.imagesOnServer[idx]))
                mainIA.originalImage = mainIA.I
                if len(mainIA.imageShape) == 3:
                    mainIA.currentViewPlane = 'z'
                else:
                    mainIA.currentViewPlane = ''
                mainIA.maskType = 'noMask'
                mainIA.labelMask = None
                mainDL.newImageFromServerStep = 0
                emit('server2ClientMessage', 'fetch plane');
                emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(mainIA.imageShape), np.min(mainIA.I), np.max(mainIA.I)))
            else:
                emit('dialog','index out of bounds')
        else:
            emit('dialog','please enter one of the provided indices')
    elif mainDL.saveCurrentImageToServerStep == 1:
        if len(message) > 0:
            if message[0].isalpha():
                if message.isalnum():
                    path = pathjoin(mainIA.dataSubfolder, '%s.tif' % message)
                    tifwrite(np.uint8(255*normalize(mainIA.I)),path)
                    emit('dialog', 'saved to server as %s.tif' % message)
                    mainDL.saveCurrentImageToServerStep = 0
                else:
                    emit('dialog', 'not all chars are alphanumeric')
            else:
                emit('dialog','first char should be a letter')
        else:
            emit('dialog', 'name should have at least one char')
    elif mainDL.saveAnnotationsToServerStep == 1:
        if len(message) > 0:
            if message[0].isalpha():
                if message.isalnum():
                    path = pathjoin(mainIA.annotationsSubfolder, '%s' % message)
                    mainIA.saveAnnotations(path)
                    emit('dialog', 'annotations saved under %s' % message)
                    mainDL.saveAnnotationsToServerStep = 0
                else:
                    emit('dialog', 'not all chars are alphanumeric')
            else:
                emit('dialog','first char should be a letter')
        else:
            emit('dialog', 'name should have at least one char')
    elif mainDL.loadAnnotationsFromServerStep == 1:
        if len(message) > 0 and message.isnumeric():
            idx = int(message)
            if idx >= 1 and idx <= len(mainIA.annotationSetsOnServer):
                mainIA.loadAnnotations(mainIA.annotationSetsOnServer[idx-1])
                [p,n,e] = fileparts(mainIA.annotationSetsOnServer[idx-1])
                emit('dialog', 'annotations '+n+' loaded')
                emit('server2ClientMessage', 'did annotate images')
                mainDL.loadAnnotationsFromServerStep = 0
            else:
                emit('dialog', 'invalid index')
        else:
            emit('dialog', 'invalid input')
    elif mainDL.editAnnotationsStep == 1:
        if len(message) > 0 and message.isnumeric():
            idx = int(message)
            if idx >= 1 and idx <= (len(mainIA.unsavedAnnotationsOnServer)/2):
                fplist = listfiles(mainIA.annotationsSubfolder,'_Img.tif')
                
                # sorting according to last index to find nClasses
                # naming convention: Image_'class index'_'image index'_Img.tif
                iSort = np.argsort([int((fplist[i].split('/')[-1]).split('_')[2]) for i in range(len(fplist))])
                mainIA.nClasses = int((fplist[iSort[idx-1]].split('/')[-1]).split('_')[1])

                I = tifread(pathjoin(mainIA.annotationsSubfolder, 'Image_%03d_%03d_Img.tif' % (mainIA.nClasses,idx-1)))
                A = tifread(pathjoin(mainIA.annotationsSubfolder, 'Image_%03d_%03d_Ant.tif' % (mainIA.nClasses,idx-1)))
                mainIA.setImage(I)
                mainIA.originalImage = mainIA.I
                if len(mainIA.imageShape) == 3:
                    mainIA.currentViewPlane = 'z'
                else:
                    mainIA.currentViewPlane = ''
                mainIA.maskType = 'labelMask'
                mainIA.labelMask = A
                mainIA.labelMaskIndex = idx-1
                if mainIA.interpolationMask is not None: # prepare for interpolation
                    mainIA.interpolationMask = np.zeros(A.shape, dtype=A.dtype)
                mainDL.editAnnotationsStep = 0
                emit('server2ClientMessage', 'fetch plane')
                emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(mainIA.imageShape), np.min(mainIA.I), np.max(mainIA.I)))
                emit('server2ClientMessage', 'nclasses %d' % mainIA.nClasses)
                emit('server2ClientMessage', 'load annotation tool')
            else:
                emit('dialog', 'invalid index')
        else:
            emit('dialog', 'invalid input')
    elif mainDL.saveMLModelToServerStep == 1:
        if len(message) > 0:
            if message[0].isalpha():
                if message.isalnum():
                    path = pathjoin(mainIA.modelsSubfolder, '%s.ml' % message)
                    mainIA.saveMLModel(path)
                    emit('dialog', 'model saved as %s.ml' % message)
                    mainDL.saveMLModelToServerStep = 0
                else:
                    emit('dialog', 'not all chars are alphanumeric')
            else:
                emit('dialog','first char should be a letter')
        else:
            emit('dialog', 'name should have at least one char')
    elif mainDL.loadMLModelFromServerStep == 1:
        if len(message) > 0 and message.isnumeric():
            idx = int(message)
            if idx >= 1 and idx <= len(mainIA.mlModelsOnServer):
                mainIA.loadMLModel(mainIA.mlModelsOnServer[idx-1])
                emit('dialog', 'model loaded')
                emit('server2ClientMessage', 'did train ml segmenter')
                emit('server2ClientMessage', 'nclasses %d' % mainIA.mlSegmenterModel['nClasses'])
                mainDL.loadMLModelFromServerStep = 0
            else:
                emit('dialog', 'invalid index')
        else:
            emit('dialog', 'invalid input')
    elif mainDL.ipStep == 1: # image processing step
        if len(message) > 0 and message.isnumeric():
            didSucceed = True
            prm = int(message)
            if any(mainDL.ipOperation == ipOp for ipOp in ['median filter','maximum filter','minimum filter']):
                emit('dialog','computing...')
                print('computing...')
                mainIA.imageProcessing(mainDL.ipOperation,{'size': prm})
                print('done')
            elif any(mainDL.ipOperation == ipOp for ipOp in ['blur','log','gradient magnitude','derivatives']):
                emit('dialog','computing...')
                mainIA.imageProcessing(mainDL.ipOperation,{'sigma': prm})
                print('done')
            elif mainDL.ipOperation == 'extract plane':
                if prm >= 1 and prm <= mainIA.nPlanes:
                    mainIA.imageProcessing(mainDL.ipOperation,{'plane': prm})
                else:
                    didSucceed = False
                    emit('dialog', 'index out of bounds')
            elif mainDL.ipOperation == 'extract channel':
                if prm >= 1 and prm <= mainIA.nChannels:
                    mainIA.imageProcessing(mainDL.ipOperation,{'channel': prm})
                else:
                    didSucceed = False
                    emit('dialog', 'index out of bounds')
            if didSucceed:
                mainDL.ipStep = 0
                emit('server2ClientMessage', 'fetch plane')
                emit('dialog', 'operation '+mainDL.ipOperation+' done')
        else:
            emit('dialog', 'invalid input')
    else:
        emit('dialog', 'invalid input')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs/<string:page_name>.html')
def renderStatic(page_name):
    return render_template('/docs/%s.html' % page_name)

@app.route('/jsonecho')
def jsonEcho():
    message = request.args.get('message','',type=str) # id, defaults, type
    return jsonify(message=message)

@app.route('/jsonupload', methods=['POST'])
def jsonUpload():
    data = request.form['text']
    parsed = json.loads(data)
    if parsed['description'] == 'annotation':
        parsedContent = parsed['content']
        labelMaskContent = parsedContent[0]
        labelMaskForInterpContent = parsedContent[1]
        mainIA.updateLabelMask(labelMaskContent,labelMaskForInterpContent)
        if len(mainIA.imageShape) == 2:
            mainIA.saveLabelMask(mainIA.annotationsSubfolder)
            mainIA.maskType = 'noMask'
        return jsonify(message='annotations uploaded')
    elif parsed['description'] == 'mlsegtrainprmts': # ML segmentation training parameters
        mainIA.mlSegmenterTrainParameters = parsed['content']
        # training procedure actually started by client after receiving following message
        return jsonify(message='training ml segmenter | this might take a while | a message will be shown here when training is done')
    elif parsed['description'] == 'mlsegprmts': # ml segmentation parameters
        mlsegprmts = parsed['content']
        pmIdx = mlsegprmts[0]-1
        segBlr,segThr = mlsegprmts[1:3]
        mainIA.mlSegment(pmIdx,segBlr/100,segThr/100)
        mainIA.maskType = 'segmMask'
        return jsonifyCurrentPlane()
    elif parsed['description'] == 'thrsegprmts': # threshold segmentation parameters
        thrsegprmts = parsed['content']
        cIdx = thrsegprmts[0]-1 # 0: dark, 1: bright
        segBlr,segThr = thrsegprmts[1:3]
        mainIA.thresholdSegment(cIdx,segBlr/100,segThr/100)
        mainIA.maskType = 'segmMask'
        return jsonifyCurrentPlane()
    elif parsed['description'] == 'spotdetprmts': # spot detection parameters
        sdSigma, sdThr = parsed['content']
        mainIA.findSpots(sdSigma,sdThr)
        return jsonify(mainIA.coords)
    elif parsed['description'] == 'plnfrmsvr': # plane from server; content: [time,channel,plane]
        mainIA.planeIndex = parsed['content'][2]
        mainIA.channelIndex = parsed['content'][1]
        # time index not supported yet
        return jsonifyCurrentPlane()
    elif parsed['description'] == 'fetchpln':
        return jsonifyCurrentPlane()
    elif parsed['description'] == 'fetchim':
        imPath = pathjoin(mainIA.scratchSubfolder,'Scratch.tif')
        tifwrite(np.uint8(255*normalize(mainIA.I)),imPath)
        return send_file(imPath,mimetype='image/tiff')
    elif parsed['description'] == 'fetchdt':
        l = mainIA.unsavedAnnotationsOnServer
        folderPath = pathjoin(mainIA.scratchSubfolder,'Annotations')
        removeFolderIfExistent(folderPath)
        createFolderIfNonExistent(folderPath)
        for f in l: copyFile(f,folderPath)
        shutil.make_archive(folderPath, 'zip', folderPath)
        return send_file(pathjoin(mainIA.scratchSubfolder,'Annotations.zip'),mimetype='application/octet-stream')
    elif parsed['description'] == 'fetchtb':
        tbPath = pathjoin(mainIA.scratchSubfolder,'Scratch.csv')
        return send_file(tbPath,mimetype='text/csv')
        
@app.route('/jsonfileupload', methods=['POST'])
def jsonFileUpload():
    if 'file' not in request.files:
        print('No file part')
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
    I = tifread(file)
    mainIA.setImage(I)
    mainIA.originalImage = mainIA.I
    if len(mainIA.imageShape) == 3:
        mainIA.currentViewPlane = 'z'
    else:
        mainIA.currentViewPlane = ''
    mainIA.maskType = 'noMask'
    mainIA.labelMask = None
    return jsonifyCurrentPlane()


if __name__ == '__main__':
    if devMode:
        # deploys to the same machine (typically at localhost:5000; see terminal log)
        socketio.run(app, debug=True)
    else:
        # deploys to local wifi network
        # client should point to IP address of server in local network
        # ip can be found on Settings (Ubuntu) or System Preferences (Mac) app
        # accessible at http://<local ip address>:5000/
        socketio.run(app, debug=False, host='0.0.0.0')
