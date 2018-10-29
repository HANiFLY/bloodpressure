# -*- coding: utf-8 -*-
import numpy as np

def LoadFile(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = [int(float(i)) for i in line[0:-1].split(',')]
            data.append(line)
    return data

def crosssetoutput():
    data1 = LoadFile('train2.csv')
    data2 = LoadFile('test2.csv')
    data = data1 + data2
    num_totalset = len(data) #前4个数据集大小
    size_of_set = num_totalset/5

    size_of_restset = num_totalset - size_of_set #最后一个数据集大小
    crossdataset = [] #初始化
    for i in range(0, 4):
        crossdataset.append([])
        while len(crossdataset[i]) != size_of_set:
            random = int(np.random.random() * len(data))
            crossdataset[i].append(data[random])
            del(data[random])
    crossdataset.append(data)
    trainset1 = crossdataset[1]+crossdataset[2]+crossdataset[3]+crossdataset[4]
    testset1 = crossdataset[0]
    trainset2 = crossdataset[0]+crossdataset[2]+crossdataset[3]+crossdataset[4]
    testset2 = crossdataset[1]
    trainset3 = crossdataset[0]+crossdataset[1]+crossdataset[3]+crossdataset[4]
    testset3 = crossdataset[2]
    trainset4 = crossdataset[0]+crossdataset[1]+crossdataset[2]+crossdataset[4]
    testset4 = crossdataset[3]
    trainset5 = crossdataset[0]+crossdataset[1]+crossdataset[2]+crossdataset[3]
    testset5 = crossdataset[4]
    return trainset1,trainset2,trainset3,trainset4,trainset5,testset1,testset2,testset3,testset4,testset5
