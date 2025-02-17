import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# Show the origin data
print(data.head())
print(data.describe())

# Plot origin data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Scatter plot of training data")
plt.show()

data.insert(0, 'Ones', 1)
# Add a column for matrix operate
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
Y = data.iloc[:, cols - 1:cols]
print(X.head())
print(Y.head())

X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0]))


def cost_function(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))


def gradient_descent(X, Y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        # Prediction error
        error = X * theta.T - Y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = temp[0, j] - alpha / len(X) * np.sum(term)

        theta = temp
        cost[i] = cost_function(X, Y, theta)

    return theta, cost


# Set learning rate and times of iterations
alpha = 0.01
iters = 1000
theta, cost = gradient_descent(X, Y, theta, alpha, iters)

print(f"Final cost: {cost}")

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + theta[0, 1] * x

plt.figure(figsize=(12, 8))
plt.xlabel("Population")
plt.ylabel("Profit")
l1 = plt.plot(x, f, label='Prediction', color='red')
l2 = plt.scatter(data.Population, data.Profit, label='Training Data')
plt.legend(loc='best')
plt.title("Predicted Profit")
plt.show()


# 画出cost的走势
plt.figure(figsize=(12, 8))
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost in training')
plt.plot(np.arange(iters), cost, 'r')
plt.show()