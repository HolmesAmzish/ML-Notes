import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

##
# Read the data and plot

path = 'ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
data.head()
data.describe()
data.plot(kind = 'scatter', x = 'Population', y = 'Profit', figsize = (12, 8))
plt.show()

data.insert(0, 'Ones', 1)
data.head()
cols = data.shape[1]
X = data.iloc[:, 0: cols - 1]
Y = data.iloc[:, cols - 1 : cols]
X.head()
Y.head()
X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0]))

X.shape, Y.shape, theta.shape

def computeCost(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))

computeCost(X, Y, theta)

# Gradient descence
def gradientDescent(X, Y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = X * theta.T - Y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = temp[0, j] - alpha / len(X) * np.sum(term)

        theta = temp
        cost[i] = computeCost(X, Y, theta)

    return theta, cost

alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, Y, theta, alpha, iters)
# 计算训练模型的误差
computeCost(X, Y, g)

# 画出拟合图像
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + g[0, 1] * x

plt.figure(figsize=(12, 8))
plt.xlabel('Population')
plt.ylabel('Profit')
l1 = plt.plot(x, f, label='Prediction', color='red')
l2 = plt.scatter(data.Population, data.Profit, label='Traing Data', )
plt.legend(loc='best')
plt.title('Predicted Profit vs Population Size')
plt.show()

# Plot the cost value

plt.figure(figsize=(12, 8))
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Error vs Training Epoch')
plt.plot(np.arange(iters), cost, 'r')
plt.show()

os.system("pause")