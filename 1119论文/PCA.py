#coding=utf-8

from numpy import *
def PCA(dataSet,k):  #dataSet（n*m）每一列代表一个样本，每一行代表一个属性，其中k代表降到k维，即lowDDateMat（k*m）
    meanData = mean(dataSet, axis = 1)
    meanData = matrix(meanData)
    meanData = meanData.transpose()
    meanArray = dataSet - meanData   #预处理，将样本均值变为0
    covMat = cov(meanArray)  #协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # 对特征值eigVals从小到大排序
    eigValInd = eigValInd[:-(k + 1):-1]  # 从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    redEigVects = eigVects[:, eigValInd]  # 返回排序后特征值对应的特征向量redEigVects（主成分）
    lowDDataMat = redEigVects.transpose()*meanArray  # 将原始数据投影到主成分上得到新的低维数据lowDDataMat
    return lowDDataMat


