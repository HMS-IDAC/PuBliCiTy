organelle = 'Mit' # 'ER', 'LD', 'Mit'

if organelle == 'ER':
    restoreVariables = True
    train = False
    test = False
    deploy = True
    
    deployResolution = 'full'

    deployImagePathIn = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop2.tif'
    deployImagePathOut = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop2_ER_CropAndSegm.tif'
    #deployFolderPathIn = ''#'E:/Jeeyun/FIB-SEM/110117_HPF_KO45min'
    #deployFolderPathOut = ''#'E:/Jeeyun/FIB-SEM/AnalysisOutputs/KO45min/ER'
    deployFolderPathIn = ''#E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1_Planes'
    deployFolderPathOut = ''#E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1_Planes_ER'

    if deploy == True:
        imSize = 120
        batchSize = 8
    if train == True or test == True:
        imSize = 60
        batchSize = 32

    modelPathIn = 'E:/Jeeyun/FIB-SEM/Models/modelERVC.ckpt'
    modelPathOut ='E:/Jeeyun/FIB-SEM/Models/modelERVC2.ckpt'

    reSplitTrainSet = True
    trainSetSplitPath = 'E:/Jeeyun/FIB-SEM/Models/modelERVC_Shuffle.data'

    logDir = 'E:/Jeeyun/FIB-SEM/Logs/logERVC2'
    logPath = 'E:/Jeeyun/FIB-SEM/Logs/imERVC2.png'
    
    imPath = 'E:/Jeeyun/FIB-SEM/TrainData/ER/TrainSet_60_Agg_CB'
    imPathTest = 'E:/Jeeyun/FIB-SEM/TestData'

    dsm = 0.6479827655767296
    dss = 0.08869744364209528

    maxBrig = 0.5
    maxCont = 0.1
    
    nFeatMapsList = [64,32,64] # length should be 3 for input 60 to have output 20
    
    learningRate = 0.00001

    nEpochs = 20
    
elif organelle == 'LD':
    restoreVariables = True
    train = False
    test = False
    deploy = True
    
    deployResolution = 'full'

    deployImagePathIn = ''#E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1.tif'
    deployImagePathOut = ''#E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1_LD.tif'
    deployFolderPathIn = 'E:/Jeeyun/FIB-SEM/110117_HPF_KO45min'
    deployFolderPathOut = 'E:/Jeeyun/FIB-SEM/AnalysisOutputs/KO45min/LD'
    #deployFolderPathIn = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1_Planes'
    #deployFolderPathOut = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1_Planes_LD'

    if deploy == True:
        imSize = 120
        batchSize = 8
    if train == True or test == True:
        imSize = 60
        batchSize = 32

    modelPathIn = 'E:/Jeeyun/FIB-SEM/Models/modelLDVC.ckpt'
    modelPathOut ='E:/Jeeyun/FIB-SEM/Models/modelLDVC2.ckpt'

    reSplitTrainSet = True
    trainSetSplitPath = 'E:/Jeeyun/FIB-SEM/Models/modelLDVC_Shuffle.data'

    logDir = 'E:/Jeeyun/FIB-SEM/Logs/logLDVC2'
    logPath = 'E:/Jeeyun/FIB-SEM/Logs/imLDVC2.png'
    
    imPath = 'E:/Jeeyun/FIB-SEM/TrainData/LD/TrainSet_60_Agg_CB'
    imPathTest = 'E:/Jeeyun/FIB-SEM/TestData'

    dsm = 0.68
    dss = 0.09

    maxBrig = 0.5
    maxCont = 0.1
    
    nFeatMapsList = [16,32,64] # length should be 3 for input 60 to have output 20
    
    learningRate = 0.00001

    nEpochs = 20
    
elif organelle == 'Mit':
    restoreVariables = True
    train = False
    test = False
    deploy = True
    
    deployResolution = 'half'

    deployImagePathIn = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop3.tif'
    deployImagePathOut = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop3_Mit_Model2.tif'
    deployFolderPathIn = ''#E:/Jeeyun/FIB-SEM/110117_HPF_KO45min'
    deployFolderPathOut = ''#E:/Jeeyun/FIB-SEM/AnalysisOutputs/KO45min/Mit'
    #deployFolderPathIn = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1_Planes'
    #deployFolderPathOut = 'E:/Jeeyun/FIB-SEM/Crops/KO45min_Crop1_Planes_Mit'

    if deploy == True:
        imSize = 60
        batchSize = 64
    if train == True or test == True:
        imSize = 60
        batchSize = 32

    modelPathIn = 'E:/Jeeyun/FIB-SEM/Models/modelMitVC2.ckpt'
    modelPathOut ='E:/Jeeyun/FIB-SEM/Models/modelMitVC3.ckpt'

    reSplitTrainSet = False
    trainSetSplitPath = 'E:/Jeeyun/FIB-SEM/Models/modelMitVC_Shuffle.data'

    logDir = 'E:/Jeeyun/FIB-SEM/Logs/logMitVC3'
    logPath = 'E:/Jeeyun/FIB-SEM/Logs/imMitVC3.png'
    
    imPath = 'E:/Jeeyun/FIB-SEM/TrainData/Mit/Downsize2_TrainSet_60_Agg_PlusCrop3Blocks_CB'
    imPathTest = 'E:/Jeeyun/FIB-SEM/TestData2'

    dsm = 0.67
    dss = 0.1

    maxBrig = 0.5
    maxCont = 0.5
    
    nFeatMapsList = [16,32,64] # length should be 3 for input 60 to have output 20
    
    learningRate = 0.0001

    nEpochs = 20


# --------------------------------------------------


import numpy as np
import tensorflow as tf
import os, shutil, sys

from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, MaxPooling3D, Flatten, concatenate, Cropping3D, Activation, Dropout
from tensorflow.keras import Input, Model

os.environ['CUDA_VISIBLE_DEVICES']='0'

import sys
from toolbox.imtools import *
from toolbox.ftools import *
from toolbox.PartitionOfImageVC import PI3D

nClasses = 2
nChannels = 1
bn = True

nImages = len(listfiles(imPath,'_Img.tif'))
nImagesTrain = int(0.9*nImages); nImagesValid = nImages-nImagesTrain

if train:
    if reSplitTrainSet:
        shuffle = np.random.permutation(nImages)
        saveData(shuffle, trainSetSplitPath)
    else:
        shuffle = loadData(trainSetSplitPath)


nImagesTest = len(listfiles(imPathTest,'_Img.tif'))

def getBatch(n, dataset='train'):
    x_batch = np.zeros((n,imSize,imSize,imSize,nChannels))
    y_batch = np.zeros((n,imSize,imSize,imSize,nClasses))

    if dataset == 'train':
        perm = np.random.permutation(nImagesTrain)
    elif dataset == 'valid':
        perm = nImagesTrain+np.random.permutation(nImagesValid)
    for i in range(n):
        I = im2double(tifread(pathjoin(imPath,'I%05d_Img.tif' % shuffle[perm[i]])))
        A = tifread(pathjoin(imPath,'I%05d_Ant.tif' % shuffle[perm[i]]))

        I = (I-dsm)/dss
#         fBrig = maxBrig*np.float_power(-1,np.random.rand() < 0.5)*np.random.rand()
#         fCont = 1+maxCont*np.float_power(-1,np.random.rand() < 0.5)*np.random.rand()
#         I = I*fCont+fBrig
        x_batch[i,:,:,:,0] = I
        for j in range(nClasses):
            y_batch[i,:,:,:,j] = A == (j+1)

    return x_batch, y_batch


# https://lmb.informatik.uni-freiburg.de/Publications/2019/FMBCAMBBR19/paper-U-Net.pdf

x = Input((imSize,imSize,imSize,nChannels))
t = tf.placeholder(tf.bool)
ccidx = []

hidden = [tf.to_float(x)]
if organelle == 'ER' and bn == True:
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
print('layer',len(hidden)-1,':',hidden[-1].shape,'input')

# down

# bna = []
for i in range(len(nFeatMapsList)-1):
    print('...')
    # if i == 0 and bn == True:
    #     hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    nFeatMaps = nFeatMapsList[i]
    hidden.append(Conv3D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    if bn:
        hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(Conv3D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    if bn:
        hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(Activation('relu')(hidden[-1]))
    # hidden.append(tf.nn.batch_normalization(hidden[-1], bnm[len(bna)], bns[len(bna)], 0.0, 1.0, 0.000001))
    # hidden.append((hidden[-1]-bnm[len(bna)])/bns[len(bna)])
    # bna.append(hidden[-1])
    # print('len bna',len(bna))
    ccidx.append(len(hidden)-1)
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after conv conv bn')
    if organelle == 'Mit':
        hidden.append(Conv3D(nFeatMaps,(2),padding='valid',activation=None,strides=2)(hidden[-1]))
    else:
        hidden.append(MaxPooling3D()(hidden[-1]))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after maxp')
    
#     hidden.append(Dropout(0.25)(hidden[-1], training=t))

# bottom

i = len(nFeatMapsList)-1
print('...')
nFeatMaps = nFeatMapsList[i]
hidden.append(Conv3D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
if bn:
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
hidden.append(Conv3D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
if bn:
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
    hidden.append(Conv3DTranspose(nFeatMaps,(3),strides=(2),padding='same',activation='relu')(hidden[-1]))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after upconv')
    toCrop = int((hidden[ccidx[-1-i]].shape[1]-hidden[-1].shape[1])//2)
    hidden.append(concatenate([hidden[-1], Cropping3D(toCrop)(hidden[ccidx[-1-i]])]))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after concat with cropped layer %d' % ccidx[-1-i])
    
#     hidden.append(Dropout(0.5)(hidden[-1], training=t))
    
    hidden.append(Conv3D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    if bn:
        hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(Conv3D(nFeatMaps,(3),padding='valid',activation=None)(hidden[-1]))
    if bn:
        hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(Activation('relu')(hidden[-1]))
    # hidden.append(tf.nn.batch_normalization(hidden[-1], bnm[len(bna)], bns[len(bna)], 0.0, 1.0, 0.000001))
    # hidden.append((hidden[-1]-bnm[len(bna)])/bns[len(bna)])
    # bna.append(hidden[-1])
    # print('len bna',len(bna))
    print('layer',len(hidden)-1,':',hidden[-1].shape,'after conv conv bn')
    print('...')

# last

hidden.append(Conv3D(nClasses,(1),padding='same',activation='softmax')(hidden[-1]))
print('layer',len(hidden)-1,':',hidden[-1].shape,'output')


# sys.exit(0)

sm = hidden[-1]
y0 = Input((imSize,imSize,imSize,nClasses))
toCrop = int((y0.shape[1]-sm.shape[1])//2)
y = Cropping3D(toCrop)(y0)
cropSize = y.shape[1]

l = []
# nl = []
for iClass in range(nClasses):
    labels0 = tf.reshape(tf.to_int32(tf.slice(y,[0,0,0,0,iClass],[-1,-1,-1,-1,1])),[batchSize,cropSize,cropSize,cropSize])
    predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(sm,4),iClass)),[batchSize,cropSize,cropSize,cropSize])
    correct = tf.multiply(labels0,predict0)
    nCorrect0 = tf.reduce_sum(correct)
    nLabels0 = tf.reduce_sum(labels0)
    l.append(tf.to_float(nCorrect0)/tf.to_float(nLabels0))
    # nl.append(nLabels0)
acc = tf.add_n(l)/nClasses

loss = -tf.reduce_sum(tf.multiply(y,tf.log(sm)))
updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learningRate)
# optimizer = tf.train.MomentumOptimizer(0.00001,0.9)
if bn:
    with tf.control_dependencies(updateOps):
        optOp = optimizer.minimize(loss)
    # optOp = optimizer.minimize(loss)
    # optOp = tf.group([optOp, updateOps])
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/batch_normalization
    # https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad
else:
    optOp = optimizer.minimize(loss)


if train:
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('acc', acc)
	mergedsmr = tf.summary.merge_all()

	# logDir = '/scratch/Jeeyun/Logs/logERVC2'

	if os.path.exists(logDir):
		shutil.rmtree(logDir)
	writer = tf.summary.FileWriter(logDir+'/train')
	writer2 = tf.summary.FileWriter(logDir+'/valid')


sess = tf.Session()
saver = tf.train.Saver()


if restoreVariables:
    saver.restore(sess, modelPathIn)
else:
    sess.run(tf.global_variables_initializer())


def testOnImage(index,bnFlag):
    V = im2double(tifread(pathjoin(imPathTest,'I%05d_Img.tif' % index)))
    if deployResolution == 'half':
        V = imresize3Double(V,[V.shape[0]/2,V.shape[1]/2,V.shape[2]/2])
    
    V = (V-dsm)/dss

    margin = 20
    PI3D.setup(V,imSize,margin)
    PI3D.createOutput(1)
    nImages = PI3D.NumPatches

    x_batch = np.zeros((batchSize,imSize,imSize,imSize,nChannels))

    print('test on image',index,'bn',bnFlag)
    for i in range(nImages):
        # print(i,nImages)
        P = PI3D.getPatch(i)
        
        j = np.mod(i,batchSize)
        x_batch[j,:,:,:,0] = P
        
        if j == batchSize-1 or i == nImages-1:
            if bn:
                output = sess.run(sm,feed_dict={x: x_batch, t: bnFlag})
            else:
                output = sess.run(sm,feed_dict={x: x_batch})
            for k in range(j+1):
                PI3D.patchOutput(i-j+k,output[k,:,:,:,0])

    PM = PI3D.Output

    V = V[20:-20,20:-20,20:-20]
    PM = PM[20:-20,20:-20,20:-20]

    return np.uint8(255*np.concatenate((normalize(V),PM),axis=2))


if train:
    ma = 0.5
    for i in range(nEpochs*nImages):
        x_batch, y_batch = getBatch(batchSize)

        if bn:
            smr,vLoss,a,_ = sess.run([mergedsmr,loss,acc,optOp],feed_dict={x: x_batch, y0: y_batch, t: True})
        else:
            smr,vLoss,a,_ = sess.run([mergedsmr,loss,acc,optOp],feed_dict={x: x_batch, y0: y_batch})
        writer.add_summary(smr, i)
        print('step', '%05d' % i, 'epoch', '%05d' % int(i/nImages), 'acc', '%.02f' % a, 'loss', '%.02f' % vLoss, 'n 0', int(np.sum(y_batch[:,:,:,:,0])), 'n 1', int(np.sum(y_batch[:,:,:,:,1])))
        
        if i % 10 == 0:
            x_batch, y_batch = getBatch(batchSize, dataset='valid')
            smr2,vLoss2,a2,_ = sess.run([mergedsmr,loss,acc,optOp],feed_dict={x: x_batch, y0: y_batch, t: False})
            writer2.add_summary(smr2, i)
            print('...step', '%05d' % i, 'acc v', '%.02f' % a2, 'loss v', '%.02f' % vLoss2, 'n 0', int(np.sum(y_batch[:,:,:,:,0])), 'n 1', int(np.sum(y_batch[:,:,:,:,1])))
        
            ma = 0.5*ma+0.5*a2
            if ma > 0.9999:
                saver.save(sess, modelPathOut)
                break

            if i % 100 == 0:
                imIndex = np.random.randint(nImagesTest)
                outBN0 = testOnImage(imIndex,False)

                imwrite(np.squeeze(np.max(outBN0,axis=0)), logPath)

                if a > 0.7:
                    saver.save(sess, modelPathOut)
                    print('model saved')

    writer.close()
    writer2.close()


if test:
    for imIndex in range(nImagesTest):
        print('testing on image', imIndex)
        outBN0 = testOnImage(imIndex,False)
        tifwrite(outBN0, pathjoin(imPathTest, 'I%05d_PM.tif' % imIndex))


def v2pm(V):
    V = (V-dsm)/dss

    margin = 20
    PI3D.setup(V,imSize,margin)
    PI3D.createOutput(1)
    nImages = PI3D.NumPatches

    x_batch = np.zeros((batchSize,imSize,imSize,imSize,nChannels))

    for i in range(nImages):
        if i % int(nImages/10) == 0:
            print(int(i/(nImages/10)),'/ 10 done')
        
        P = PI3D.getPatch(i)

        j = np.mod(i,batchSize)
        x_batch[j,:,:,:,0] = P

        if j == batchSize-1 or i == nImages-1:
            if bn:
                output = sess.run(sm,feed_dict={x: x_batch, t: False})
            else:
                output = sess.run(sm,feed_dict={x: x_batch})
            for k in range(j+1):
                PI3D.patchOutput(i-j+k,output[k,:,:,:,0])

    PM = PI3D.Output
    return PM


if deploy:
    if deployImagePathIn != '':
        V = im2double(tifread(deployImagePathIn))
        if deployResolution == 'half':
            Vshape = V.shape
            V = imresize3Double(V,[V.shape[0]/2,V.shape[1]/2,V.shape[2]/2])

        V = (V-dsm)/dss

        margin = 20
        PI3D.setup(V,imSize,margin)
        PI3D.createOutput(1)
        nImages = PI3D.NumPatches

        x_batch = np.zeros((batchSize,imSize,imSize,imSize,nChannels))

        for i in range(nImages):
            print('processing block', i, 'of', nImages)
            P = PI3D.getPatch(i)

            j = np.mod(i,batchSize)
            x_batch[j,:,:,:,0] = P

            if j == batchSize-1 or i == nImages-1:
                if bn:
                    output = sess.run(sm,feed_dict={x: x_batch, t: False})
                else:
                    output = sess.run(sm,feed_dict={x: x_batch})
                for k in range(j+1):
                    PI3D.patchOutput(i-j+k,output[k,:,:,:,0])

        PM = PI3D.Output
        if deployResolution == 'half':
            PM = imresize3Double(PM,[Vshape[0],Vshape[1],Vshape[2]])
        
        tifwrite(np.uint8(255*PM), deployImagePathOut)



    if deployFolderPathIn != '':
        paths = listsubdirs(deployFolderPathIn)
        imNames = []
        for path in paths:
            l = listfiles(path,'Cryo_','._Cryo')
            print(len(l))
            imNames += l
        
        # imNames = listfiles(deployFolderPathIn, '.tif')

        print('found', len(imNames), 'images')
			
        I = tifread(imNames[0])
        nPlanes = len(imNames)
        
        margin = 20
        patchSize = imSize
        if deployResolution == 'half':
            margin *= 2
            patchSize *= 2
            
        subPatchSize = patchSize-2*margin
        nSlices = int(np.floor((nPlanes-2*margin)/subPatchSize))
        if nSlices*subPatchSize+2*margin < nPlanes:
            nSlices += 1
                    
        for iZ in range(0, nSlices):
            z0 = np.minimum(iZ*subPatchSize,nPlanes-patchSize)
            z1 = z0+patchSize
            sz0 = z0+margin
            sz1 = sz0+subPatchSize
            print(z0,sz0,sz1,z1)

            V = np.zeros((patchSize,I.shape[0],I.shape[1]))
            if deployResolution == 'half':
                Vshape = V.shape
            for i in range(z0,z1):
                print('processing slice', iZ, 'of', nSlices, 'reading plane', i)
                V[i-z0,:,:] = im2double(tifread(imNames[i]))

            if deployResolution == 'half':
                V = imresize3Double(V,[V.shape[0]/2,V.shape[1]/2,V.shape[2]/2])
                
            print('v2pm...')
            PM = v2pm(V)
            print('...done')
            
            if deployResolution == 'half':
                PM = imresize3Double(PM,[Vshape[0],Vshape[1],Vshape[2]])

            print('writing...')
            for i in range(sz0,sz1):
                print('writing plane', i)
                tifwrite(np.uint8(255*PM[i-sz0+margin,:,:]), pathjoin(deployFolderPathOut, 'I_%05d.tif' % i))
            print('...done')

        # remaining margin planes are zero
        I0 = np.zeros(I.shape,dtype=np.uint8)
        for i in range(margin):
            print('writing plane', i)
            tifwrite(I0, pathjoin(deployFolderPathOut, 'I_%05d.tif' % i))
        for i in range(sz1,nPlanes):
            print('writing plane', i)
            tifwrite(I0, pathjoin(deployFolderPathOut, 'I_%05d.tif' % i))


sess.close()