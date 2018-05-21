import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from ResnetModels import *
from DatasetGenerator import DatasetGenerator


# --------------------------------------------------------------------------------

class DMLTrainer ():

    # ---- Train the densenet network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- nnClassCount - number of output classes
    # ---- trBatchSize - batch size
    # ---- trMaxEpoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    # ---- checkpoint - if not None loads the model and continues training

    def train(self, pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize,
              trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):
        # -------------------- SETTINGS: NETWORK ARCHITECTURE

        model_1 = ResNet18(nnClassCount, nnIsTrained).cuda()
        model_2 = DenseNet121(nnClassCount, nnIsTrained).cuda()

        model_1 = torch.nn.DataParallel(model_1).cuda()
        model_2 = torch.nn.DataParallel(model_2).cuda()

        # -------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence = transforms.Compose(transformList)

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain,
                                        transform=transformSequence)
        datasetVal = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal,
                                      transform=transformSequence)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=24,
                                     pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24,
                                   pin_memory=True)

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer_1 = optim.Adam(model_1.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler_1 = ReduceLROnPlateau(optimizer_1, factor=0.1, patience=5, mode='min')

        optimizer_2 = optim.Adam(model_2.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler_2 = ReduceLROnPlateau(optimizer_2, factor=0.1, patience=5, mode='min')
        # -------------------- SETTINGS: LOSS
        loss_CE = torch.nn.BCELoss(size_average=False)
        loss_KLD = torch.nn.KLDivLoss(size_average=False)

        # ---- Load checkpoint
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model_1.load_state_dict(modelCheckpoint['state_dict'])
            optimizer_1.load_state_dict(modelCheckpoint['optimizer_1'])
            optimizer_2.load_state_dict(modelCheckpoint['optimizer_2'])
        # ---- TRAIN THE NETWORK

        lossMIN_1 = 100000
        lossMIN_2 = 100000

        for epochID in range(0, trMaxEpoch):

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            self.epochTrain(model_1, model_2, dataLoaderTrain, optimizer_1, optimizer_2, scheduler_1, scheduler_2, trMaxEpoch, nnClassCount, loss_CE, loss_KLD)

            lossVal_1, losstensor_1 = self.epochVal(model_1, model_2, dataLoaderVal, optimizer_1, scheduler_1, trMaxEpoch, nnClassCount, loss_CE, loss_KLD)
            lossVal_2, losstensor_2 = self.epochVal(model_2, model_1, dataLoaderVal, optimizer_2, scheduler_2, trMaxEpoch, nnClassCount, loss_CE, loss_KLD)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler_1.step(losstensor_1.data[0])
            scheduler_2.step(losstensor_2.data[0])
            if lossVal_1 < lossMIN_1:
                lossMIN_1 = lossVal_1
                torch.save({'epoch': epochID + 1, 'state_dict': model_1.state_dict(), 'best_loss': lossMIN_1,
                            'optimizer': optimizer_1.state_dict()}, 'm1-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] los1_1= ' + str(lossVal_1))
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss_1= ' + str(lossVal_1))
            if epochID == trMaxEpoch:
                torch.save({'epoch': epochID + 1, 'state_dict': model_1.state_dict(), 'best_loss': lossMIN_1,
                            'optimizer': optimizer_1.state_dict()}, 'm1-final-checkpoint' + '.pth.tar')


            if lossVal_2 < lossMIN_2:
                lossMIN_2 = lossVal_2
                torch.save({'epoch': epochID + 1, 'state_dict': model_2.state_dict(), 'best_loss': lossMIN_2,
                            'optimizer': optimizer_2.state_dict()}, 'm2-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss_2= ' + str(lossVal_2))
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss_2= ' + str(lossVal_2))
            if epochID == trMaxEpoch:
                torch.save({'epoch': epochID + 1, 'state_dict': model_2.state_dict(), 'best_loss': lossMIN_1,
                            'optimizer': optimizer_2.state_dict()}, 'm2-final-checkpoint' + '.pth.tar')

    # --------------------------------------------------------------------------------

    def epochTrain(self, model_1, model_2, dataLoader, optimizer_1, optimizer_2, scheduler_1, scheduler_2, epochMax, classCount, loss_CE, loss_KLD):

        model_1.train()
        print('Training!')
        for batchID, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput= torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            #varTarget = varTarget.astype(np.int64)
            #varTarget = varTarget.type(torch.cuda.LongTensor)

            varOutput_1 = model_1(varInput)
            varOutput_2 = model_2(varInput)
            Output_2 = torch.autograd.Variable(varOutput_2.data, volatile=True)
            lossvalue_1 = loss_CE(varOutput_1, varTarget) + loss_KLD(varOutput_1, Output_2)

            optimizer_1.zero_grad()
            lossvalue_1.backward()
            optimizer_1.step()

            varOutput_1 = model_1(varInput)
            varOutput_2 = model_2(varInput)

            Output_1 = torch.autograd.Variable(varOutput_1.data, volatile=True)
            lossvalue_2 = loss_CE(varOutput_2, varTarget) + loss_KLD(varOutput_2, Output_1)

            optimizer_2.zero_grad()
            lossvalue_2.backward()
            optimizer_2.step()

        print('finish one times')
    # --------------------------------------------------------------------------------


    def epochVal(self, model_1, model_2, dataLoader, optimizer, scheduler, epochMax, classCount,  loss_CE, loss_KLD):

        model_1.eval()
        model_2.eval()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput = torch.autograd.Variable(input, volatile=True)
            varTarget = torch.autograd.Variable(target, volatile=True)
            #varTarget = varTarget.type(torch.cuda.LongTensor)

            varOutput_1 = model_1(varInput)
            varOutput_2 = model_2(varInput)
            Output_2 = torch.autograd.Variable(varOutput_2.data, volatile=True)
            losstensor = loss_CE(varOutput_1, varTarget) + loss_KLD(varOutput_1, Output_2)
            losstensorMean += losstensor

            lossVal += losstensor.data[0]
            lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean

    # --------------------------------------------------------------------------------

    # ---- Computes area under ROC curve
    # ---- dataGT - ground truth data
    # ---- dataPRED - predicted data
    # ---- classCount - number of classes

    def computeAUROC(self, dataGT, dataPRED, classCount):

        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

        return outAUROC

    # --------------------------------------------------------------------------------

    # ---- Test the trained network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- nnClassCount - number of output classes
    # ---- trBatchSize - batch size
    # ---- trMaxEpoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    # ---- checkpoint - if not None loads the model and continues training

    def test(self, pathDirData, pathFileTest, pathModel_1, pathModel_2, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize,
             transResize, transCrop):

        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                       'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                       'Hernia']

        cudnn.benchmark = True

        # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        model_1 = ResNet18(nnClassCount, nnIsTrained).cuda()
        model_2 = DenseNet121(nnClassCount, nnIsTrained).cuda()

        model_1 = torch.nn.DataParallel(model_1).cuda()
        model_2 = torch.nn.DataParallel(model_2).cuda()

        modelCheckpoint_1 = torch.load(pathModel_1)
        modelCheckpoint_2 = torch.load(pathModel_2)
        model_1.load_state_dict(modelCheckpoint_1['state_dict'])
        model_2.load_state_dict(modelCheckpoint_2['state_dict'])

        # -------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # -------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest,
                                       transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False,
                                    pin_memory=True)

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        model_1.eval()

        for i, (input, target) in enumerate(dataLoaderTest):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)

            bs, n_crops, c, h, w = input.size()

            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)

            out = model_1(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)
            outPRED = torch.cat((outPRED, outMean.data), 0)

        #       aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocIndividual = self.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()

        print('AUROC mean ', aurocMean)

        for i in range(0, len(aurocIndividual)):
            print(CLASS_NAMES[i], ' ', aurocIndividual[i])

        model_2.eval()

        for i, (input, target) in enumerate(dataLoaderTest):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)

            bs, n_crops, c, h, w = input.size()

            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)

            out = model_2(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)
            outPRED = torch.cat((outPRED, outMean.data), 0)

        #       aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocIndividual = self.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()

        print('AUROC mean ', aurocMean)

        for i in range(0, len(aurocIndividual)):
            print(CLASS_NAMES[i], ' ', aurocIndividual[i])

        return
# --------------------------------------------------------------------------------





