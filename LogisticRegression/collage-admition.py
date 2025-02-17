import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'ex2data1.txt'
data = pd.read_csv(path, names=['exam1', 'exam2', 'admitted'])

print(data.head())
print(data.describe())

positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', alpha=0.5, label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', alpha=0.5, label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, Y):
    first = Y * np.log(sigmoid(X @ theta.T))
    second = (1 - Y) * np.log(1 - sigmoid(X @ theta.T))
    return -1 * np.mean(first + second)

def gradient(theta, X, Y):
    return (1 / len(X) * X.T @ (sigmoid(X @ theta.T) - Y))

data.insert(0, 'Ones', 1)
X = data.iloc[:, 0: -1].values
Y = data.iloc[:, -1].values
theta = np.zeros(3)

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
res = opt.minimize(fun=cost, x0=np.array(theta), args=(X, np.array(Y)), method='Newton-CG', jac=gradient)
print(res)

coef = -res.x / res.x[2]
x = np.arange(30, 100, 0.5)
y = coef[0] + coef[1] * x

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', alpha=0.5, label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', alpha=0.5, label='Not Admitted')
ax.plot(x, y, label='Decision Boundary', c='grey')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()