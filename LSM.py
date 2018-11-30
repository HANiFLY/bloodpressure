# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import loadfile
import partition
import csv
from scipy.optimize import leastsq



def func(p,x):
    k, b = p
    return k*x+b


def error(p,x,y):
    return func(p,x) - y

data = loadfile.LoadFile('train.csv')
data = np.array(data)
index = partition.partition(data)
# print 'index',index
ptt = data[:, 4]
sbp = data[:,-2]  # 高压和低压

k_value = []
b_value = []
out = open('k_b.csv', 'wb')   #写入文件名，移植后更改此处文件名！！！！！！！！！！！！！
csv_write = csv.writer(out, dialect='excel')
csv_write.writerow(['k', 'b'])
for i in range(len(index) - 1):
    p0 = np.array([-0.35, 20])
    start = index[i]
    end = index[i+1]
    x = ptt[start:end]
    y = sbp[start: end]
    # print 'x',x
    Para = leastsq(error, p0, args=(x, y))
    k, b = Para[0]
    csv_write.writerow([k, b])
    k_value.append(round(k,2))
    b_value.append(round(b,2))

print k_value
print b_value



out.close()

# print('cost:'+str(Para[1]))
# print('拟合的直线为:')
# print("y="+str(round(k,2))+"x+"+str(round(b,2)))
