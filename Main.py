import os
import numpy as np
import time
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from ChexnetTrainer import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    
#    runTest()
    runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/home/dataset/ChextXRay'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_label.txt'
    pathFileVal = './dataset/val_label.txt'
    pathFileTest = './dataset/test_label.txt'

    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    #densenet 21/169/201     --DENSE-NET-121    --ResNet-18
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 64
    trMaxEpoch = 10

    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    Trainer = ChexnetTrainer()
    Trainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    print ('Testing the trained model')
    Trainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/home/dataset/ChextXRay'
    pathFileTest = './dataset/test_label.txt'
    '''
    nnArchitecture = 'DENSE-NET-121'
    '''
    nnArchitecture = 'ResNet-18'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = './m-final-checkpoint.pth.tar'

    timestampLaunch = ''
    Trainer = ChexnetTrainer()
    Trainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





