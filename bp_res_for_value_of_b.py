# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('gb18030')
import csv
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
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
    for i in range(len(data[0])):
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
    out_num = 1   # 数量更改，2代表高压及低压两个值
    hid_num = 15  #人为设定隐层神经元数量
    w1 = 0.2 * np.random.random((sample_len, hid_num)) - 0.1   #  6*15的数组
    w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1      # 15*1的数组
    hid_offset = np.zeros(hid_num)    #1*15的数组
    out_offset = np.zeros(out_num)    #1*1的数组
    input_learnrate = 0.2
    hid_learnrate = 0.2

    # while 1:
    error = []
    for numIter in range(100):
        for i in range(sample_num):
            t_label = label[i]
            inputVec = sample[i]
            #前向的过程
            hid_value=np.dot(inputVec,w1)+hid_offset #隐层的输入  （1*6的数组）
            hid_act=get(hid_value)                 #隐层对应的输出   （1*15的数组）
            out_value=np.dot(hid_act,w2)+out_offset        #（1*2的数组）
            out_act=get(out_value)    #输出层最后的输出  （1*2的数组）

            #后向过程
            err=t_label-out_act  #（1*2的数组）
            err= float(err)
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
            error.append(abs(err))
 #   print 'error is', np.matrix(error).transpose()
 #   print len(error)
 #    plot(error[0:5000])
    return w1, w2, hid_offset, out_offset


def Test(k1, k2):
    # time_start = time.time()
    data = LoadFile('generated_data.csv')
    data = np.array(data)
    data = data.astype(float)
    data[:, -2] = data[:, -2] - k1 * data[:, 4] - k2 * data[:, -3]
    data = np.delete(data, 4, axis=1) #去除ptt
    data = np.delete(data, -3, axis=1) #去除心率
    data = np.delete(data, -1, axis=1) #去除低压
    # print data
    data = data_norm(data)
    train_data = data[:, 0:-1] #4个参数
    train_label = data[:, -1] #高压


    data = LoadFile('test.csv')
    data = np.array(data)
    data = data.astype(float)
    data[:,-2] = data[:,-2]- k1*data[:,4]- k2 * data[:, -3]
    bmin = min(data[:, -2])#高压中最小值
    # print 'bmin',bmin
    bmax = max(data[:, -2])#高压中最大值

    data = np.delete(data, 4, axis=1)
    data = np.delete(data, -3, axis=1)
    data = np.delete(data, -1, axis=1)
    data = data_norm(data)
    test_data = data[:, 0:-1]
    test_label = data[:,-1]

    RMSE = 0
    tn = len(data)
    w1, w2, hid_offset, out_offset = TrainNetwork(train_data,train_label)
    # print tn
    # out = open('b_value.csv', 'wb')   #写入文件名，移植后更改此处文件名！！！！！！！！！！！！！
    # csv_write = csv.writer(out, dialect='excel')
    # csv_write.writerow(['b_true', 'b_estimated','error'])
    output_error = []
    for i in range(0, tn):
        hid_value=np.dot(test_data[i], w1)+hid_offset
        hid_act=get(hid_value)
        out_value=np.dot(hid_act,w2)+out_offset
        out_act=get(out_value)
        out_act = float(out_act)
        output_error.append(abs((test_label[i] - out_act) * (bmax -bmin)))
        # print test_label[i] * (bmax - bmin) + bmin, out_act * (bmax - bmin) + bmin, (test_label[i] - out_act) * (bmax -bmin)
        RMSE += ((test_label[i] - out_act) * (bmax -bmin))**2
        # RMSE1 += ((test_label[i][0] - out_act[0])*(sbpmax - sbpmin))**2
        # RMSE2 += ((test_label[i][1] - out_act[1])*(dbpmax - dbpmin))**2
        # RMSE += RMSE1 + RMSE2

    #     csv_write.writerow([test_label[i] * (bmax - bmin) + bmin, out_act * (bmax - bmin) + bmin, (test_label[i] - out_act) * (bmax -bmin)])
    # out.close()
    # print (RMSE1/tn)**0.5
    # print (RMSE2/tn)**0.5
    accurate = correct = incorrect = mistake = 0
    for i in range(len(output_error)):
        if output_error[i] <= 5:
            accurate += 1
        elif output_error[i] <=10:
            correct += 1
        elif output_error[i] <= 15:
            incorrect += 1
        else: mistake += 1
    # print accurate, correct, incorrect, mistake


    return (RMSE/tn)**0.5

    # time_end = time.time()
    # print('totally cost', time_end - time_start)
step = 0.1
i = -2
x1 = []
y1 = []
while(i<=2):
    x1.append(i)
    y1.append(Test(0.15,i))
    i += step
print x1
print y1
# plt.subplot(1,2,1)
plt.plot(x1,y1)
plt.xlabel('k2_value')
plt.ylabel('RMSE_value')



# j = -2
# x2 = []
# y2 = []
# while(j<=2):
#     x2.append(j)
#     y2.append(Test(j,0.5))
#     j += step
# print 'x2',x2
# print y2
# # plt.subplot(1,2,2)
# plt.plot(x2,y2)
# plt.xlabel('k1_value')
# plt.ylabel('RMSE_value')
plt.show()

