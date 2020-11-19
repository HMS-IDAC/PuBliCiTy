restoreVariables = True
train = False
test = True
deploy = False

deployImagePathIn = '/home/cicconet/Development/PuBliCiTy/DataForPC/Test/I00000_Img.tif'
deployImagePathOut = '/home/cicconet/Download/I00000_PM.tif'
deployFolderPathIn = ''
deployFolderPathOut = ''

imSize = 60
nClasses = 3
nChannels = 1
batchSize = 32

modelPathIn = '/home/cicconet/Development/PuBliCiTy/Models/unet2D_v0.ckpt'
modelPathOut ='/home/cicconet/Development/PuBliCiTy/Models/unet2D_v0.ckpt'

reSplitTrainSet = True
trainSetSplitPath = '/home/cicconet/Development/PuBliCiTy/Models/trainSetSplit.data'

logDir = '/home/cicconet/Development/PuBliCiTy/Logs/unet2D'
logPath = '/home/cicconet/Development/PuBliCiTy/Logs/unet2D_TestSample.tif'

imPath = '/home/cicconet/Development/PuBliCiTy/DataForPC/Train_60'
imPathTest = '/home/cicconet/Development/PuBliCiTy/DataForPC/Test'

nFeatMapsList = [16,32,64] # length should be 3 for input 60 to have output 20

learningRate = 0.00001

nEpochs = 20



import numpy as np
import os, shutil, sys

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Flatten, concatenate, Cropping2D, Activation
from tensorflow.keras import Input, Model

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
t = tf.placeholder(tf.bool)
ccidx = []

hidden = [tf.to_float(x)]
print('layer',len(hidden)-1,':',hidden[-1].shape,'input')

# nFeatMapsList = [16,32,64] # length should be 3 for input 60 to have output 20
# nFeatMapsList = [16,32,64,128] # length should be 4 for input 108 to have output 20

# down

for i in range(len(nFeatMapsList)-1):
    print('...')
    nFeatMaps = nFeatMapsList[i]
    hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
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
hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
hidden.append(Activation('relu')(hidden[-1]))
# hidden.append(tf.nn.batch_normalization(hidden[-1], bnm[len(bna)], bns[len(bna)], 0.0, 1.0, 0.000001))
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
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(Conv2D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(Activation('relu')(hidden[-1]))
    # hidden.append(tf.nn.batch_normalization(hidden[-1], bnm[len(bna)], bns[len(bna)], 0.0, 1.0, 0.000001))
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
    labels0 = tf.reshape(tf.to_int32(tf.slice(y,[0,0,0,iClass],[-1,-1,-1,1])),[batchSize,cropSize,cropSize])
    predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(sm,3),iClass)),[batchSize,cropSize,cropSize])
    correct = tf.multiply(labels0,predict0)
    nCorrect0 = tf.reduce_sum(correct)
    nLabels0 = tf.reduce_sum(labels0)
    l.append(tf.to_float(nCorrect0)/tf.to_float(nLabels0))
    # nl.append(nLabels0)
acc = tf.add_n(l)/nClasses

loss = -tf.reduce_sum(tf.multiply(y,tf.log(sm)))
updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learningRate)
with tf.control_dependencies(updateOps):
    optOp = optimizer.minimize(loss)


if train:
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    mergedsmr = tf.summary.merge_all()

    if os.path.exists(logDir):
        shutil.rmtree(logDir)

    writer = tf.summary.FileWriter(pathjoin(logDir, 'train'))
    writer2 = tf.summary.FileWriter(pathjoin(logDir, 'valid'))


sess = tf.Session()
saver = tf.train.Saver()

if restoreVariables:
    saver.restore(sess, modelPathIn)
else:
    sess.run(tf.global_variables_initializer())


def testOnImage(index,bnFlag):
    V = im2double(tifread(pathjoin(imPathTest,'I%05d_Img.tif' % index)))

    margin = 20
    # margin = 44
    PI2D.setup(V,imSize,margin)
    PI2D.createOutput(nClasses)
    nPatches = PI2D.NumPatches

    x_batch = np.zeros((batchSize,imSize,imSize,nChannels))

    print('test on image',index)
    for i in range(nPatches):
        P = PI2D.getPatch(i)
        
        j = np.mod(i,batchSize)
        if nChannels == 1:
            x_batch[j,:,:,0] = P
        else:
            for k in range(nChannels):
                x_batch[j,:,:,k] = P[k,:,:]
        
        if j == batchSize-1 or i == nPatches-1:
            output = sess.run(sm,feed_dict={x: x_batch, t: bnFlag})
            for k in range(j+1):
                PI2D.patchOutput(i-j+k,np.moveaxis(output[k,:,:,0:nClasses],[2,0,1],[0,1,2]))

    PM = PI2D.Output
    if nChannels == 1:
        W = V[margin:-margin,margin:-margin]
    else:
        W = np.mean(V[:,margin:-margin,margin:-margin],axis=0)

    W = normalize(W)

    for i in range(PM.shape[0]):
        PMi = PM[i,margin:-margin,margin:-margin]
        W = np.concatenate((W, PMi), axis=1)

    return np.uint8(255*W)

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
                pred = testOnImage(imIndex,False)

                tifwrite(pred, logPath)

                if a > 0.7:
                    saver.save(sess, modelPathOut)
                    print('model saved')

    writer.close()
    writer2.close()

if test:
    for imIndex in range(nImagesTest):
        pred = testOnImage(imIndex,False)
        tifwrite(pred, pathjoin(imPathTest, 'Pred_I%05d.tif' % imIndex))

if deploy:
    for imIndex in range(13):
        p = '/home/mc457/files/CellBiology/IDAC/Marcelo/Hotamisligil/Parlakgul/EM3D/Planes4Annotation/All3C3_Img/I%05d_Img.tif' % imIndex

        V = im2double(tifread(p))
        # V = (V-dsm)/dss

        # margin = 20
        margin = 44
        PI2D.setup(V,imSize,margin)
        PI2D.createOutput(3)
        nImages = PI2D.NumPatches

        x_batch = np.zeros((batchSize,imSize,imSize,nChannels))

        for i in range(nImages):
            print(imIndex,i,nImages)
            P = PI2D.getPatch(i)
            
            j = np.mod(i,batchSize)
            for k in range(nChannels):
                x_batch[j,:,:,k] = P[k,:,:]
            
            if j == batchSize-1 or i == nImages-1:
                output = sess.run(sm,feed_dict={x: x_batch, t: False})
                for k in range(j+1):
                    PI2D.patchOutput(i-j+k,np.moveaxis(output[k,:,:,0:3],[2,0,1],[0,1,2]))

        PM = PI2D.Output
        # PM0 = PM[0,margin:-margin,margin:-margin]
        # PM1 = PM[1,margin:-margin,margin:-margin]
        # V0 = normalize(V[1,margin:-margin,margin:-margin])

        # PM0 = PM[0,:,:]
        # PM1 = PM[1,:,:]
        V0 = V[1,:,:]
        tifwrite(255*np.uint8(V0),'/scratch/Gunes/Predict/I%05d_Im.tif' % imIndex)

        A = np.argmax(PM,axis=0)
        S = np.sum(PM,axis=0)
        A[S == 0] = -1
        for i in range(2):
            tifwrite(255*np.uint8(A == i),'/scratch/Gunes/Predict/I%05d_P%d.tif' % (imIndex,i))

        # C = np.zeros((V0.shape[0],V0.shape[1],3))
        # C[:,:,0] = V0
        # C[:,:,1] = V0
        # C[:,:,2] = V0
        # image1 = np.uint8(255*C)
        # tifwrite(image1,'/scratch/Gunes/Predict/I%05d_Im.tif' % imIndex)

        # C[:,:,0] = imgaussfilt(PM0,2)
        # C[:,:,1] = imgaussfilt(PM1,2)
        # C[:,:,2] = 0
        # image2 = np.uint8(255*C)
        # tifwrite(image2,'/scratch/Gunes/Predict/I%05d_PM.tif' % imIndex)

        # B = blendRGBImages(image1,image2)
        # tifwrite(B,'/scratch/Gunes/Predict/I%05d.tif' % imIndex)


sess.close()