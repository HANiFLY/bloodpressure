# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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


data = LoadFile('train.csv')
data = np.array(data)
height = data[:,2]
data = data_norm(data)


#数据分割（有哪几个人）
index = [0]
i = 0; j = 1
while j <len(height):
    if height[j] == height[i]:
        j += 1
    else:
        index.append(j)
        i = j
        j += 1
index.append(len(height))
print index
print len(index)
corrcoe_sbp = []
corrcoe_dbp = []
for i in range(len(index) - 1):
    start_index = index[i]
    end_index = index[i+1]
    sbp = data[start_index:end_index, -2]
    dbp = data[start_index:end_index, -1]
    ptt = data[start_index:end_index, 4]
    corrcoe_sbp_temp = np.corrcoef(ptt,sbp)[0][1]
    corrcoe_dbp_temp = np.corrcoef(ptt, dbp)[0][1]
    corrcoe_sbp.append(corrcoe_sbp_temp)
    corrcoe_dbp.append(corrcoe_dbp_temp)
print corrcoe_sbp
print np.mean(corrcoe_sbp)



# dataframe = pd.DataFrame({'sbp_corr_coeff':corrcoe_sbp, 'dbp_corr_coeff': corrcoe_dbp})
# dataframe.to_csv('corr_coeff_normalized',index=True, sep=',')
# print dataframe

