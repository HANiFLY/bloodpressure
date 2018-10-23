# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import random


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
    for i in range(1, 8):
        data[:, i] = (data[:, i] - min(data[:, i]))*1.0/(max(data[:, i]) - min(data[:, i]))
    return data

#定义Sigmoid函数
def get(x):
    act_vec=[]
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec=np.array(act_vec)
    return act_vec


#训练BP神经网络
def TrainNetwork(sample,label):
    sample_num = len(sample)   #样本数目
    sample_len = len(sample[0])   #属性数目

    out_num = 1
    hid_num = 15
    w1 = 0.2 * np.random.random((sample_len, hid_num)) - 0.1
    w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1
    hid_offset = np.zeros(hid_num)
    out_offset = np.zeros(out_num)
    input_learnrate = 0.2
    hid_learnrate = 0.2
    sign = 0
    ech = 1
    # while 1:
    for j in range(500):
        err = []
        for i in range(0,len(sample)):
            i = int(np.random.rand()*sample_num)
            t_label = label[i]

            #前向的过程
            hid_value=np.dot(sample[i],w1)+hid_offset #隐层的输入            
            hid_act=get(hid_value)                 #隐层对应的输出                                 
            out_value=np.dot(hid_act,w2)+out_offset
            out_act=get(out_value)    #输出层最后的输出                                 

            #后向过程
            err=t_label-out_act
            
            out_delta=err*out_act*(1-out_act) #输出层的方向梯度方向                         
            hid_delta = hid_act*(1 - hid_act) * np.dot(w2, out_delta)   
            for j in range(0,out_num):
                w2[:,j]+=hid_learnrate*out_delta[j]*hid_act
            for k in range(0,hid_num):
                w1[:,k]+=input_learnrate*hid_delta[k]*sample[i]

            out_offset += hid_learnrate * out_delta   #阈值的更新                    
            hid_offset += input_learnrate * hid_delta

            # if abs(sum(err)) < 1e-6:
            #     break
        # print abs(sum(err))
        if abs(sum(err)) < 1e-10:
            break 
    return w1,w2,hid_offset,out_offset

def Test():
    time_start = time.time()
    data = LoadFile('train.csv')

    data = np.array(data)
    
    

    data = data_norm(data)

    train_data = data[:, 0:-2] #6个参数
    train_label = data[:, -2] #高压
    train_label1 = data[:, -1] #低压

    data = LoadFile('test.csv')
    data = np.array(data)

    sbpmin = min(data[:, -2])#高压中最小值
    sbpmax = max(data[:, -2])#高压中最大值
    dbpmin = min(data[:, -1])#低压中最小值
    dbpmax = max(data[:, -1])#低压中最大值
    data = data_norm(data)

    test_data = data[:, 0:-2]
    test_label = data[:, -2]
    test_label1 = data[:, -1]

    w1,w2,hid_offset,out_offset=TrainNetwork(train_data,train_label1)

    RMSE = 0
    tn = 0
    t10 = 0
    for i in range(0,len(test_label1)):
        tn = tn + 1
        hid_value=np.dot(test_data[i],w1)+hid_offset     
        hid_act=get(hid_value)                             
        out_value=np.dot(hid_act,w2)+out_offset            
        out_act=get(out_value)                             
        print test_label1[i]*(dbpmax - dbpmin) + dbpmin, out_act[0]*(dbpmax - dbpmin) + dbpmin, (test_label1[i] - out_act[0])*(dbpmax - dbpmin)
        RMSE += (((test_label1[i] - out_act[0])*(dbpmax - dbpmin)))**2
        if abs((test_label1[i] - out_act[0])*(dbpmax - dbpmin)) > 10:
            t10 = t10 + 1
    print (RMSE/tn)**0.5, t10



    w1,w2,hid_offset,out_offset=TrainNetwork(train_data,train_label)

    tn = 0
    t15 = 0
    RMSE = 0
    for i in range(0,len(test_label)):
        tn = tn + 1
        hid_value=np.dot(test_data[i],w1)+hid_offset     
        hid_act=get(hid_value)                             
        out_value=np.dot(hid_act,w2)+out_offset            
        out_act=get(out_value)                             
        print test_label[i]*(sbpmax - sbpmin) + sbpmin, out_act[0]*(sbpmax - sbpmin) + sbpmin, (test_label[i] - out_act[0])*(sbpmax - sbpmin)
        RMSE += (((test_label[i] - out_act[0])*(sbpmax - sbpmin)))**2
    
        if abs((test_label[i] - out_act[0])*(sbpmax - sbpmin)) > 15:
            t15 = t15 + 1
    print (RMSE/tn)**0.5, t15

    time_end = time.time()
    print('totally cost', time_end - time_start)
Test()