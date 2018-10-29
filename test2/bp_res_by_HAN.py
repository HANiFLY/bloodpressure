# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('gb18030')

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import crosscheck
from pylab import *

#定义数据文件读取函数
def LoadFile(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = [int(float(i)) for i in line[0:-1].split(',')]
            data.append(line)
    return data

def data_norm(data):
    data = data.astype(float)
    for i in range(1, 7):
        data[:, i] = (data[:, i] - min(data[:, i]))*1.0/(max(data[:, i]) - min(data[:, i]))
    return data

#定义Sigmoid函数
def get(x):
    act_vec=[]
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec=np.array(act_vec)
    return act_vec


def plot(data_to_draw):
    length = len(data_to_draw)
    x = range(length)
    plt.plot(x, data_to_draw)
    plt.show()

#训练BP神经网络
def TrainNetwork(sample,label):
    sample_num = len(sample)   #样本数目
    sample_len = len(sample[0])   #属性数目
    out_num = 2   # 数量更改，2代表高压及低压两个值
    hid_num = 15  #人为设定隐层神经元数量
    w1 = 0.2 * np.random.random((sample_len, hid_num)) - 0.1   #  6*15的数组
    w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1      # 15*2的数组
    hid_offset = np.zeros(hid_num)    #1*15的数组
    out_offset = np.zeros(out_num)    #1*2的数组
    input_learnrate = 0.2
    hid_learnrate = 0.2

    # while 1:
    error = []
    for numIter in range(200):
        for i in range(sample_num):
            t_label = label[i]       #随机选取一个输入
            inputVec = sample[i]
            #前向的过程
            hid_value=np.dot(inputVec,w1)+hid_offset #隐层的输入  （1*6的数组）
            hid_act=get(hid_value)                 #隐层对应的输出   （1*15的数组）
            out_value=np.dot(hid_act,w2)+out_offset        #（1*2的数组）
            out_act=get(out_value)    #输出层最后的输出  （1*2的数组）

            #后向过程
            err=t_label-out_act  #（1*2的数组）

            out_delta=err*out_act*(1-out_act) #输出层的方向梯度方向    （1*2的数组）
            trans_out_w2 = np.array(np.matrix(w2).transpose())   #  2*15的数组
            hid_delta = hid_act * (1 - hid_act) * np.dot(out_delta, trans_out_w2)     # 1*15的数组

            trans_hid_act = np.array(np.matrix(hid_act).transpose())   #(15*1的数组)，便于点积计算
            trans_input = np.array(np.matrix(inputVec).transpose())

            #更新权重及阈值
            w2 += hid_learnrate * np.array(np.matrix(trans_hid_act) * np.matrix(out_delta)) #隐层与输出层间权重的更新
            out_offset -= hid_learnrate * out_delta    #输出层阈值的更新
            w1 += input_learnrate * np.array(np.matrix(trans_input) * np.matrix(hid_delta))  # 输入层与隐层间权重的更新
            hid_offset -= input_learnrate * hid_delta   #隐层阈值的更新
            error.append(err[0]**2+err[1]**2)
 #   print 'error is', np.matrix(error).transpose()
 #   print len(error)
  #  plot(error[0:10000])
    return w1, w2, hid_offset, out_offset

trainset1,trainset2,trainset3,trainset4,trainset5,testset1,testset2,testset3,testset4,testset5 = crosscheck.crosssetoutput()

def Test1():
    #time_start = time.time()
    #data = LoadFile('train2.csv')
    data = trainset1
    data = np.array(data)
    data = data_norm(data)
    train_data = data[:, 0:-2] #6个参数
    train_label = data[:, -2:] #高压和低压

    #data = LoadFile('test2.csv')
    data = testset1
    data = np.array(data)

    sbpmin = min(data[:, -2])#高压中最小值
    sbpmax = max(data[:, -2])#高压中最大值
    dbpmin = min(data[:, -1])#低压中最小值
    dbpmax = max(data[:, -1])#低压中最大值
    data = data_norm(data)

    test_data = data[:, 0:-2]
    test_label = data[:, -2:]

    RMSE = 0
    RMSE1 = RMSE2 = 0
    tn = len(data)

    w1, w2, hid_offset, out_offset = TrainNetwork(train_data,train_label)

    for i in range(0, tn):
        hid_value=np.dot(test_data[i], w1)+hid_offset
        hid_act=get(hid_value)
        out_value=np.dot(hid_act,w2)+out_offset
        out_act=get(out_value)
        #print test_label[i][1] * (dbpmax - dbpmin) + dbpmin, out_act[1] * (dbpmax - dbpmin) + dbpmin, (test_label[i][1] - out_act[1]) * (dbpmax - dbpmin)
        #print test_label[i][0] * (sbpmax - sbpmin) + sbpmin, out_act[0] * (sbpmax - sbpmin) + sbpmin, (test_label[i][0] - out_act[0]) * (sbpmax - sbpmin)
        RMSE1 += ((test_label[i][0] - out_act[0])*(sbpmax - sbpmin))**2
        RMSE2 += ((test_label[i][1] - out_act[1])*(dbpmax - dbpmin))**2
        RMSE += RMSE1 + RMSE2

    print (RMSE1/tn)**0.5
    print (RMSE2/tn)**0.5

    #time_end = time.time()
    #print('totally cost', time_end - time_start)


#################################
#test2
def Test2():
    #time_start = time.time()
    #data = LoadFile('train2.csv')
    data = trainset2
    data = np.array(data)
    data = data_norm(data)
    train_data = data[:, 0:-2] #6个参数
    train_label = data[:, -2:] #高压和低压

    #data = LoadFile('test2.csv')
    data = testset2
    data = np.array(data)

    sbpmin = min(data[:, -2])#高压中最小值
    sbpmax = max(data[:, -2])#高压中最大值
    dbpmin = min(data[:, -1])#低压中最小值
    dbpmax = max(data[:, -1])#低压中最大值
    data = data_norm(data)

    test_data = data[:, 0:-2]
    test_label = data[:, -2:]

    RMSE = 0
    RMSE1 = RMSE2 = 0
    tn = len(data)

    w1, w2, hid_offset, out_offset = TrainNetwork(train_data,train_label)

    for i in range(0, tn):
        hid_value=np.dot(test_data[i], w1)+hid_offset
        hid_act=get(hid_value)
        out_value=np.dot(hid_act,w2)+out_offset
        out_act=get(out_value)
        #print test_label[i][1] * (dbpmax - dbpmin) + dbpmin, out_act[1] * (dbpmax - dbpmin) + dbpmin, (test_label[i][1] - out_act[1]) * (dbpmax - dbpmin)
        #print test_label[i][0] * (sbpmax - sbpmin) + sbpmin, out_act[0] * (sbpmax - sbpmin) + sbpmin, (test_label[i][0] - out_act[0]) * (sbpmax - sbpmin)
        RMSE1 += ((test_label[i][0] - out_act[0])*(sbpmax - sbpmin))**2
        RMSE2 += ((test_label[i][1] - out_act[1])*(dbpmax - dbpmin))**2
        RMSE += RMSE1 + RMSE2

    print (RMSE1/tn)**0.5
    print (RMSE2/tn)**0.5

    #time_end = time.time()
    #print('totally cost', time_end - time_start)



##################################################################
def Test3():
    #time_start = time.time()
    #data = LoadFile('train2.csv')
    data = trainset3
    data = np.array(data)
    data = data_norm(data)
    train_data = data[:, 0:-2] #6个参数
    train_label = data[:, -2:] #高压和低压

    #data = LoadFile('test2.csv')
    data = testset3
    data = np.array(data)

    sbpmin = min(data[:, -2])#高压中最小值
    sbpmax = max(data[:, -2])#高压中最大值
    dbpmin = min(data[:, -1])#低压中最小值
    dbpmax = max(data[:, -1])#低压中最大值
    data = data_norm(data)

    test_data = data[:, 0:-2]
    test_label = data[:, -2:]

    RMSE = 0
    RMSE1 = RMSE2 = 0
    tn = len(data)

    w1, w2, hid_offset, out_offset = TrainNetwork(train_data,train_label)

    for i in range(0, tn):
        hid_value=np.dot(test_data[i], w1)+hid_offset
        hid_act=get(hid_value)
        out_value=np.dot(hid_act,w2)+out_offset
        out_act=get(out_value)
        #print test_label[i][1] * (dbpmax - dbpmin) + dbpmin, out_act[1] * (dbpmax - dbpmin) + dbpmin, (test_label[i][1] - out_act[1]) * (dbpmax - dbpmin)
        #print test_label[i][0] * (sbpmax - sbpmin) + sbpmin, out_act[0] * (sbpmax - sbpmin) + sbpmin, (test_label[i][0] - out_act[0]) * (sbpmax - sbpmin)
        RMSE1 += ((test_label[i][0] - out_act[0])*(sbpmax - sbpmin))**2
        RMSE2 += ((test_label[i][1] - out_act[1])*(dbpmax - dbpmin))**2
        RMSE += RMSE1 + RMSE2

    print (RMSE1/tn)**0.5
    print (RMSE2/tn)**0.5

    #time_end = time.time()
    #print('totally cost', time_end - time_start)




def Test4():
    #time_start = time.time()
    #data = LoadFile('train2.csv')
    data = trainset4
    data = np.array(data)
    data = data_norm(data)
    train_data = data[:, 0:-2] #6个参数
    train_label = data[:, -2:] #高压和低压

    #data = LoadFile('test2.csv')
    data = testset4
    data = np.array(data)

    sbpmin = min(data[:, -2])#高压中最小值
    sbpmax = max(data[:, -2])#高压中最大值
    dbpmin = min(data[:, -1])#低压中最小值
    dbpmax = max(data[:, -1])#低压中最大值
    data = data_norm(data)

    test_data = data[:, 0:-2]
    test_label = data[:, -2:]

    RMSE = 0
    RMSE1 = RMSE2 = 0
    tn = len(data)

    w1, w2, hid_offset, out_offset = TrainNetwork(train_data,train_label)

    for i in range(0, tn):
        hid_value=np.dot(test_data[i], w1)+hid_offset
        hid_act=get(hid_value)
        out_value=np.dot(hid_act,w2)+out_offset
        out_act=get(out_value)
        #print test_label[i][1] * (dbpmax - dbpmin) + dbpmin, out_act[1] * (dbpmax - dbpmin) + dbpmin, (test_label[i][1] - out_act[1]) * (dbpmax - dbpmin)
        #print test_label[i][0] * (sbpmax - sbpmin) + sbpmin, out_act[0] * (sbpmax - sbpmin) + sbpmin, (test_label[i][0] - out_act[0]) * (sbpmax - sbpmin)
        RMSE1 += ((test_label[i][0] - out_act[0])*(sbpmax - sbpmin))**2
        RMSE2 += ((test_label[i][1] - out_act[1])*(dbpmax - dbpmin))**2
        RMSE += RMSE1 + RMSE2

    print (RMSE1/tn)**0.5
    print (RMSE2/tn)**0.5

    #time_end = time.time()
    #print('totally cost', time_end - time_start)



def Test5():
    #time_start = time.time()
    #data = LoadFile('train2.csv')
    data = trainset5
    data = np.array(data)
    data = data_norm(data)
    train_data = data[:, 0:-2] #6个参数
    train_label = data[:, -2:] #高压和低压

    #data = LoadFile('test2.csv')
    data = testset5
    data = np.array(data)

    sbpmin = min(data[:, -2])#高压中最小值
    sbpmax = max(data[:, -2])#高压中最大值
    dbpmin = min(data[:, -1])#低压中最小值
    dbpmax = max(data[:, -1])#低压中最大值
    data = data_norm(data)

    test_data = data[:, 0:-2]
    test_label = data[:, -2:]

    RMSE = 0
    RMSE1 = RMSE2 = 0
    tn = len(data)

    w1, w2, hid_offset, out_offset = TrainNetwork(train_data,train_label)

    for i in range(0, tn):
        hid_value=np.dot(test_data[i], w1)+hid_offset
        hid_act=get(hid_value)
        out_value=np.dot(hid_act,w2)+out_offset
        out_act=get(out_value)
        #print test_label[i][1] * (dbpmax - dbpmin) + dbpmin, out_act[1] * (dbpmax - dbpmin) + dbpmin, (test_label[i][1] - out_act[1]) * (dbpmax - dbpmin)
        #print test_label[i][0] * (sbpmax - sbpmin) + sbpmin, out_act[0] * (sbpmax - sbpmin) + sbpmin, (test_label[i][0] - out_act[0]) * (sbpmax - sbpmin)
        RMSE1 += ((test_label[i][0] - out_act[0])*(sbpmax - sbpmin))**2
        RMSE2 += ((test_label[i][1] - out_act[1])*(dbpmax - dbpmin))**2
        RMSE += RMSE1 + RMSE2

    print (RMSE1/tn)**0.5
    print (RMSE2/tn)**0.5

    #time_end = time.time()
    #print('totally cost', time_end - time_start)


Test1()
print '111111111111111111111'
Test2()
print '22222222222222222'
Test3()
print '3333333333333333333333'
Test4()
print '4444444444444444444'
Test5()