import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'is_admitted'])

print(data.head())

positive = data[data['is_admitted'] == 1]
negative = data[data['is_admitted'] == 0]


fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', alpha=0.5, label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', alpha=0.5, label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

X = data[['exam1', 'exam2']].values
Y = data['is_admitted'].values

model = LogisticRegression()
model.fit(X, Y)

print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

coef = model.coef_[0]
intercept = model.intercept_[0]
x = np.linspace(30, 100, 1000)
y = -(coef[0] * x + intercept) / coef[1]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', alpha=0.5, label='Admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', alpha=0.5, label='Not Admitted')
ax.plot(x, y, label='Decision Boundary', c='grey')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
