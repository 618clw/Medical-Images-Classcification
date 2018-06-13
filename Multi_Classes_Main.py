import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


from Multi_Classes_Trainer import MultiClassesTrainer


def main():
    #runTest()
    runTrain()


# --------------------------------------------------------------------------------

def runTrain():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # ---- Path to the directory with images
    pathDirData = '/home/dataset/ChextXRay'
    #pathDirData = '/home/group7/_CheXNet/after_'

    # ---- Paths to the files with training, validation and testing sets.
    # ---- Each file should contains pairs [path to image, output vector]
    # ---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain_1 = './dataset/class_1_train_label.txt'
    pathFileTrain_2 = './dataset/class_2_train_label.txt'
    pathFileTrain_3 = './dataset/class_3_train_label.txt'
    pathFileTrain_4 = './dataset/class_4_train_label.txt'
    pathFileVal_1 = './dataset/class_1_val_label.txt'
    pathFileVal_2 = './dataset/class_2_val_label.txt'
    pathFileVal_3 = './dataset/class_3_val_label.txt'
    pathFileVal_4 = './dataset/class_4_val_label.txt'
    pathFileTest_1 = './dataset/class_1_test_label.txt'
    pathFileTest_2 = './dataset/class_2_test_label.txt'
    pathFileTest_3 = './dataset/class_3_test_label.txt'
    pathFileTest_4 = './dataset/class_4_test_label.txt'


    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    # densenet 21/169/201     --DENSE-NET-121    --ResNet-18
    nnArchitecture = 'ResNet-18'
    nnIsTrained = True
    nnClassCount_1 = 6
    nnClassCount_2 = 4
    nnClassCount_3 = 2
    nnClassCount_4 = 2
    # ---- Training settings: batch size, maximum number of epochs
    trBatchSize = 64
    trMaxEpoch = 20

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel_1 = 'm_class_1-' + timestampLaunch + '.pth.tar'
    pathModel_2 = 'm_class_2-' + timestampLaunch + '.pth.tar'
    pathModel_3 = 'm_class_3-' + timestampLaunch + '.pth.tar'
    pathModel_4 = 'm_class_4-' + timestampLaunch + '.pth.tar'
    # pathModel_2 = 'm2-' + timestampLaunch + '.pth.tar'
    class_1 = '_class_1'
    class_2 = '_class_2'
    class_3 = '_class_3'
    class_4 = '_class_4'

    print('Class 1, Training NN architecture = ', nnArchitecture)
    Trainer_1 = MultiClassesTrainer()
    Trainer_1.train(pathDirData, pathFileTrain_1, pathFileVal_1, nnArchitecture, nnIsTrained, nnClassCount_1, trBatchSize,
                  trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, class_1, None)

    print('Class 1, Testing the trained model')
    Trainer_1.test(pathDirData, pathFileTest_1, pathModel_1, nnArchitecture, nnClassCount_1, nnIsTrained, trBatchSize,
                 imgtransResize, imgtransCrop)

    print('Class 2, Training NN architecture = ', nnArchitecture)
    Trainer_2 = MultiClassesTrainer()
    Trainer_2.train(pathDirData, pathFileTrain_2, pathFileVal_2, nnArchitecture, nnIsTrained, nnClassCount_2,
                    trBatchSize,
                    trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, class_2, None)

    print('Class 2, Testing the trained model')
    Trainer_2.test(pathDirData, pathFileTest_2, pathModel_2, nnArchitecture, nnClassCount_2, nnIsTrained, trBatchSize,
                   imgtransResize, imgtransCrop)

    print('Class 3, Training NN architecture = ', nnArchitecture)
    Trainer_3 = MultiClassesTrainer()
    Trainer_3.train(pathDirData, pathFileTrain_3, pathFileVal_3, nnArchitecture, nnIsTrained, nnClassCount_3,
                    trBatchSize,
                    trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, class_3, None)

    print('Class 3, Testing the trained model')
    Trainer_3.test(pathDirData, pathFileTest_3, pathModel_3, nnArchitecture, nnClassCount_3, nnIsTrained, trBatchSize,
                   imgtransResize, imgtransCrop)

    print('Class 4, Training NN architecture = ', nnArchitecture)
    Trainer_4 = MultiClassesTrainer()
    Trainer_4.train(pathDirData, pathFileTrain_4, pathFileVal_4, nnArchitecture, nnIsTrained, nnClassCount_4,
                    trBatchSize,
                    trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, class_4, None)

    print('Class 4, Testing the trained model')
    Trainer_4.test(pathDirData, pathFileTest_4, pathModel_4, nnArchitecture, nnClassCount_4, nnIsTrained, trBatchSize,
                   imgtransResize, imgtransCrop)
# --------------------------------------------------------------------------------

def runTest():
    pathDirData = '/home/dataset/ChextXRay'
    #pathDirData = '/home/group7/_CheXNet/after_'
    pathFileTest = './dataset/class_4_test_label.txt'
    '''
    nnArchitecture = 'DENSE-NET-121'
    '''
    nnArchitecture = 'ResNet-18'
    nnIsTrained = True
    nnClassCount = 2
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = './m_class_4-06062018-123339.pth.tar'
    # pathModel = './m-13052018-181616.pth.tar'

    timestampLaunch = ''
    Trainer = MultiClassesTrainer()
    Trainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize,
                 imgtransResize, imgtransCrop)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

