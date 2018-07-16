import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


data = pd.read_csv(r'E:\PycharmProjects\machine_learning\linear_regression\Folds5x2_pp.csv')
print(data.head())
print(data.shape)
X = data[['AT', 'V', 'AP', 'RH']]
print(X.head())
Y = data[['PE']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
print(X_train.shape)
print(Y_train.shape)

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, Y_train)
print(linreg.intercept_)
print(linreg.coef_)

# 模型平均
Y_pred = linreg.predict(X_test)
from sklearn import metrics
# 计算均方误差
mse = metrics.mean_squared_error(Y_test, Y_pred)
# 计算均方根误差
rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('mse: ', mse)
print('rmse: ', rmse)

from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, Y, cv=10)
print('mse1: ', metrics.mean_squared_error(Y, predicted))
print('rmse1: ', np.sqrt(metrics.mean_squared_error(Y, predicted)))

# 绘图
plt.scatter(Y, predicted, c='r')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
plt.show()
