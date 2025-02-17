import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('ex1data1.txt', header=None, names=['population', 'profit'])
print(data.head())

data.plot(kind='scatter', x='population', y='profit')
plt.show()

X = data['population'].values.reshape(-1, 1)
y = data['profit'].values

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

print(f"Linear model equations: y = {slope:.2f}x + {intercept:.2f}")

data.plot(kind='scatter', x='population', y='profit', label='Data')
plt.plot(X, model.predict(X), color='red', label='Linear Fit')
plt.legend()
plt.show()
