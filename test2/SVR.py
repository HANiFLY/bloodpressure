#coding:utf-8
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#############################################################################
# 读取数据
X_train = []
y_train1 = []
y_train2 = []
X_test = []
y_test1 = []
y_test2 = []

with open('train.csv', 'r') as f:
    lines = f.readlines()
    for l in lines:
        l = l.split(',')
        X_train.append([float(i) for i in l[0:6]])
        y_train1.append(float(l[-2]))
        y_train2.append(float(l[-1]))

with open('test.csv', 'r') as f:
    lines = f.readlines()
    for l in lines:
        l = l.split(',')
        X_test.append([float(i) for i in l[0:6]])
        y_test1.append(float(l[-2]))
        y_test2.append(float(l[-1]))


#############################################################################
# 训练SVR模型

#初始化SVR
# parameters = {'kernel':('linear', 'rbf'), 'C':[1.28, 10]}
# svr = GridSearchCV(SVR(),parameters)
svr = GridSearchCV(SVR(), cv=3, param_grid = [
                    # {'C': [100], 'kernel': ['linear']}])
                    {'C': [ 1.28 ], 'gamma': np.logspace(-42, 0, 25), 'kernel': ['rbf']}])
                   # param_grid={"C": [1e0, 1e-1, 1e-2],
                   #             "gamma": np.logspace(-25, 0, 25)})
# #记录训练时间
# t0 = time.time()
#训练
svr.fit(X_train, y_train2)
# svr_fit = time.time() - t0
print 'best params', svr.best_params_
# t0 = time.time()
#测试
y_svr = svr.predict(X_test)
# svr_predict = time.time() - t0

max_error = 0
RMSE = 0
for i in range(len(y_svr)):
    print y_test2[i], y_svr[i], y_svr[i] - y_test2[i]
    RMSE += (y_svr[i] - y_test2[i])**2
print 'RMSE is', (RMSE/len(y_svr))**0.5
print y_test2
#############################################################################  
#对结果进行显示
x = np.array(np.arange(len(y_test2)))
plt.figure()
plt.scatter(x, y_test2, c = 'r')
plt.scatter(x, y_svr, c = 'g')
# plt.plot(y_test2, c='k', label='data', zorder=1)
# plt.plot(y_svr, c='r', label='predicted data', zorder=2)
# plt.plot(y_svr, c='r',
#          label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))

plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus True Value')
plt.legend()

plt.show()