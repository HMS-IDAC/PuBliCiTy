"""
    pixel classifier via convolutional neural networks (U-Net)

    see *control panel* section of unet2D.py for instructions
"""

# if __name__ == '__main__': # indent under when building documentation

# ----------------------------------------------------------------------------------------------------
# control panel

restoreVariables = True
# if True: resume training (if train = True) from previous 'checkpoint' (stored at modelPathIn, set below)
# if False: start training (if train = True) from scratch
# to test or deploy a trained model, set restoreVariables = True

train = False
# if True, the script goes over the training steps,
# either updating a model from scratch or from a previous checkpoint;
# check portions of the code inside the 'if train:' directive for details, or to adapt the code if needed

test = True
# if True, the script runs predictions on a test set (defined by imPathTest below);
# check portions of the code inside the 'if test:' directive for details, or to adapt the code if needed

deploy = True
# if True, runs prediction either on a single image, or on a folder of images (see below);
# check portions of the code inside the 'if deploy:' directive for details, or to adapt the code if needed

deployImagePathIn = 'DataForPC/Deploy_In/I00000_Img.tif'
# full path to image to deploy on; set to empty string, '', if you want to ignore this deployment option

deployFolderPathIn = 'DataForPC/Deploy_In'
# full path to folder containing images deploy on; set to empty string, '', if you want to ignore this option

deployFolderPathOut = 'DataForPC/Deploy_Out'
# folder path where outputs of prediction (probability maps) are saved;
# the script ads _PMs to the respective input image name when naming the output

imSize = 60
# size of square image patches in the training set;
# if len(nFeatMapsList) = 3 (see below), imSize = 60 leads to a prediction of size 20

nClasses = 3
# number of pixel classes

nChannels = 1
# number of image channels

batchSize = 32
# batch size

modelPathIn = 'Models/unet2D_v0.ckpt'
# input model path to recover model from (when restoreVariables = True)

modelPathOut ='Models/unet2D_v0.ckpt'
# path where to save model

reSplitTrainSet = False
# if to re-split training set into training/validation subsets;
# this should be set to True every time the training set changes, which happens
# the first time the model is trained, when new training examples are added to the training set;
# otherwise set to false, so that the training and validation sets are consistent throughout multiple
# runs of training when restoreVariables = True

trainSetSplitPath = 'Models/trainSetSplit2D.data'
# where to save training/validation split information (only indices are saved)

logDir = 'Logs/unet2D'
# path to folder where to save data for real-time visualization during training via tensorboard

logPath = 'Logs/unet2D_TestSample.tif'
# path where to save prediction on a random image from imPathTest (see below) during training

imPath = 'DataForPC/Train_60'
# path to folder containing training/validation set;
# images should be of size nChannels x imSize x imSize, named I%05d_Img.tif,
# and having a corresponding I%05d_Ant.tif, a uint8 image of the same size,
# where pixels of class 1,2,... have intensity value 1,2,... respectivelly

imPathTest = 'DataForPC/Test'
# path to folder containing images for testing;
# the test set is assumed to contain images I00000_Img.tif, I00001_Img.tif, etc.;
# for each I%05d_Img.tif, the script saves corresponding probability maps as Pred_I%05d.tif

nFeatMapsList = [16,32,64]
# list of depth of feature maps at corresponding layer;
# length should be 3 for input 60 to have output 20

learningRate = 0.00001
# learning rate

nEpochs = 20
# number of epochs

useGPU = True
# if to use GPU or not


# ----------------------------------------------------------------------------------------------------
# machine room

import numpy as np
import os, shutil, sys

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Flatten, concatenate, Cropping2D, Activation, BatchNormalization
from tensorflow.keras import Input, Model

if useGPU:
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
else:
	os.environ['CUDA_VISIBLE_DEVICES']=''

from gpfunctions import *
from PartitionOfImageVC import PI2D

nImages = len(listfiles(imPath,'_Img.tif'))
nImagesTrain = int(0.9*nImages)
nImagesValid = nImages-nImagesTrain

if train:
    if reSplitTrainSet:
        shuffle = np.random.permutation(nImages)
        saveData(shuffle, trainSetSplitPath)
    else:
        shuffle = loadData(trainSetSplitPath)


nImagesTest = len(listfiles(imPathTest,'_Img.tif'))


def getBatch(n,dataset='train'):
    x_batch = np.zeros((n,imSize,imSize,nChannels))
    y_batch = np.zeros((n,imSize,imSize,nClasses))

    if dataset == 'train':
        perm = np.random.permutation(nImagesTrain)
    elif dataset == 'valid':
        perm = nImagesTrain+np.random.permutation(nImagesValid)

    for i in range(n):
        I = im2double(tifread(pathjoin(imPath,'I%05d_Img.tif' % shuffle[perm[i]])))
        A = tifread(pathjoin(imPath,'I%05d_Ant.tif' % shuffle[perm[i]]))

        if nChannels == 1:
            x_batch[i,:,:,0] = I
        else:
            for j in range(nChannels):
                x_batch[i,:,:,j] = I[j,:,:]
        for j in range(nClasses):
            y_batch[i,:,:,j] = A == (j+1)

    return x_batch, y_batch



# https://lmb.informatik.uni-freiburg.de/Publications/2019/FMBCAMBBR19/paper-U-Net.pdf

x = Input((imSize,imSize,nChannels))
t = tf.compat.v1.placeholder(tf.bool)
ccidx = []

hidden = [tf.cast(x, dtype=tf.float32)]
hidden.append(BatchNormalization()(hidden[-1], training=t))
print('layer',len(hidden)-1,':',hidden[-1].shape,'input')

# nFeatMapsList = [16,32,64] # length should be 3 for input 60 to have output 20
# nFeatMapsList = [16,32,64,128] # length should be 4 for input 108 to have output 20

# down

for i in range(len(nFeatMapsList)-1):
    print('...')
    nFeatMaps = nFeatMapsList[i]
    hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    hidden.append(BatchNormalization()(hidden[-1], training=t))
    hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    hidden.append(BatchNormalization()(hidden[-1], training=t))
    hidden.append(Activation('relu')(hidden[-1]))
    ccidx.append(len(hidden)-1)
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after conv conv bn')
    # hidden.append(MaxPooling2D()(hidden[-1]))
    hidden.append(Conv2D(nFeatMaps,(2),padding='valid',activation=None,strides=2)(hidden[-1]))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after downsampling')

# bottom

i = len(nFeatMapsList)-1
print('...')
nFeatMaps = nFeatMapsList[i]
hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
hidden.append(BatchNormalization()(hidden[-1], training=t))
hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
hidden.append(BatchNormalization()(hidden[-1], training=t))
hidden.append(Activation('relu')(hidden[-1]))
# hidden.appBatchNormalization()(hidden[-1], bnm[len(bna)], bns[len(bna)], 0.0, 1.0, 0.000001))
# hidden.append((hidden[-1]-bnm[len(bna)])/bns[len(bna)])
# bna.append(hidden[-1])
# print('len bna',len(bna))
print('layer',len(hidden)-1,':',hidden[-1].shape,'after conv conv bn')
print('...')

# up
for i in range(len(nFeatMapsList)-1):
    nFeatMaps = nFeatMapsList[-i-2]
    # hidden.append(Conv2DTranspose(nFeatMaps,(3),strides=(2),padding='same',activation='relu')(hidden[-1]))
    hidden.append(UpSampling2D(size=2)(hidden[-1]))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after upsampling')
    toCrop = int((hidden[ccidx[-1-i]].shape[1]-hidden[-1].shape[1])//2)
    hidden.append(concatenate([hidden[-1], Cropping2D(toCrop)(hidden[ccidx[-1-i]])]))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after concat with cropped layer %d' % ccidx[-1-i])
    hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    hidden.append(BatchNormalization()(hidden[-1], training=t))
    hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    hidden.append(BatchNormalization()(hidden[-1], training=t))
    hidden.append(Activation('relu')(hidden[-1]))
    # hidden.appBatchNormalization()(hidden[-1], bnm[len(bna)], bns[len(bna)], 0.0, 1.0, 0.000001))
    # hidden.append((hidden[-1]-bnm[len(bna)])/bns[len(bna)])
    # bna.append(hidden[-1])
    # print('len bna',len(bna))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after conv conv bn')
    print('...')

# last

hidden.append(Conv2D(nClasses,(1),padding='same',activation='softmax')(hidden[-1]))
print('layer',len(hidden)-1,':',hidden[-1].shape,'output')


sm = hidden[-1]
y0 = Input((imSize,imSize,nClasses))
toCrop = int((y0.shape[1]-sm.shape[1])//2)
y = Cropping2D(toCrop)(y0)
cropSize = y.shape[1]


l = []
# nl = []
for iClass in range(nClasses):
    labels0 = tf.reshape(tf.cast(tf.slice(y,[0,0,0,iClass],[-1,-1,-1,1]), dtype=tf.int32),[batchSize,cropSize,cropSize])
    predict0 = tf.reshape(tf.cast(tf.equal(tf.argmax(input=sm,axis=3),iClass), dtype=tf.int32),[batchSize,cropSize,cropSize])
    correct = tf.multiply(labels0,predict0)
    nCorrect0 = tf.reduce_sum(input_tensor=correct)
    nLabels0 = tf.reduce_sum(input_tensor=labels0)
    l.append(tf.cast(nCorrect0, dtype=tf.float32)/tf.cast(nLabels0, dtype=tf.float32))
    # nl.append(nLabels0)
acc = tf.add_n(l)/nClasses

loss = -tf.reduce_sum(input_tensor=tf.multiply(y,tf.math.log(sm)))
updateOps = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
optimizer = tf.compat.v1.train.AdamOptimizer(learningRate)
with tf.control_dependencies(updateOps):
    optOp = optimizer.minimize(loss)


if train:
    tf.compat.v1.summary.scalar('loss', loss)
    tf.compat.v1.summary.scalar('acc', acc)
    mergedsmr = tf.compat.v1.summary.merge_all()

    if os.path.exists(logDir):
        shutil.rmtree(logDir)

    writer = tf.compat.v1.summary.FileWriter(pathjoin(logDir, 'train'))
    writer2 = tf.compat.v1.summary.FileWriter(pathjoin(logDir, 'valid'))


sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver()

if restoreVariables:
    saver.restore(sess, modelPathIn)
else:
    sess.run(tf.compat.v1.global_variables_initializer())


def imageToProbMapsWithPI2D(I):
    margin = 20
    # margin = 44
    PI2D.setup(I,imSize,margin)
    PI2D.createOutput(nClasses)
    nPatches = PI2D.NumPatches

    x_batch = np.zeros((batchSize,imSize,imSize,nChannels))

    for i in range(nPatches):
        P = PI2D.getPatch(i)
        
        j = np.mod(i,batchSize)
        if nChannels == 1:
            x_batch[j,:,:,0] = P
        else:
            for k in range(nChannels):
                x_batch[j,:,:,k] = P[k,:,:]
        
        if j == batchSize-1 or i == nPatches-1:
            output = sess.run(sm,feed_dict={x: x_batch, t: False})
            for k in range(j+1):
                PI2D.patchOutput(i-j+k,np.moveaxis(output[k,:,:,0:nClasses],[2,0,1],[0,1,2]))

    return PI2D.Output

def testOnImage(index):
    print('test on image',index)

    I = im2double(tifread(pathjoin(imPathTest,'I%05d_Img.tif' % index)))
    PM = imageToProbMapsWithPI2D(I)

    if nChannels == 1:
        J = I#[margin:-margin,margin:-margin]
    else:
        J = np.mean(I, axis=0)
        # J = np.mean(I[:,margin:-margin,margin:-margin],axis=0)

    J = normalize(J)

    for i in range(PM.shape[0]):
        PMi = PM[i,:,:]
        # PMi = PM[i,margin:-margin,margin:-margin]
        J = np.concatenate((J, PMi), axis=1)

    return np.uint8(255*J)

if train:
    ma = 0.5
    for i in range(nEpochs*nImages):
        x_batch, y_batch = getBatch(batchSize,'train')

        smr,vLoss,a,_ = sess.run([mergedsmr,loss,acc,optOp],feed_dict={x: x_batch, y0: y_batch, t: True})
        writer.add_summary(smr, i)
        print('step', '%05d' % i, 'epoch', '%05d' % int(i/nImages), 'acc', '%.02f' % a, 'loss', '%.02f' % vLoss)
        
        if i % 10 == 0:
            x_batch, y_batch = getBatch(batchSize,'valid')

            smr2,vLoss2,a2,_ = sess.run([mergedsmr,loss,acc,optOp],feed_dict={x: x_batch, y0: y_batch, t: False})
            writer2.add_summary(smr2, i)
            print('(valid) step', '%05d' % i, 'epoch', '%05d' % int(i/nImages), 'acc', '%.02f' % a2, 'loss', '%.02f' % vLoss2)


            ma = 0.5*ma+0.5*a2
            if ma > 0.9999:
                saver.save(sess, modelPathOut)
                break

            if i % 100 == 0:
                imIndex = np.random.randint(nImagesTest)
                pred = testOnImage(imIndex)

                tifwrite(pred, logPath)

                if a > 0.7:
                    saver.save(sess, modelPathOut)
                    print('model saved')

    writer.close()
    writer2.close()

if test:
    for imIndex in range(nImagesTest):
        pred = testOnImage(imIndex)
        tifwrite(pred, pathjoin(imPathTest, 'Pred_I%05d.tif' % imIndex))

def deployOnImage(imPathIn, dirPathOut):
    I = im2double(tifread(imPathIn))
    PM = imageToProbMapsWithPI2D(I)
    PM = np.uint8(255*PM)
    [_,name,ext] = fileparts(imPathIn)
    tifwrite(PM, pathjoin(dirPathOut, name+'_PMs'+ext))

if deploy:
    if deployImagePathIn != '':
        print('deploying on image...')
        print(deployImagePathIn)
        deployOnImage(deployImagePathIn, deployFolderPathOut)

    if deployFolderPathIn != '':
        imPathList = listfiles(deployFolderPathIn, '.tif')
        for idx, imPathIn in enumerate(imPathList):
            print('deploying on image %d of %d' % (idx+1, len(imPathList)))
            deployOnImage(imPathIn, deployFolderPathOut)            


sess.close()