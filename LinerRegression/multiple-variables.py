import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('ex1data2.txt', header=None, names=['area', 'bedrooms', 'price'])
print(data.head())

X = data[['area', 'bedrooms']].values
y = data['price'].values

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_

print(f"Linear model equation: y = {coefficients[0]:.2f} * area + {coefficients[1]:.2f} * bedrooms + {intercept:.2f}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['area'], data['bedrooms'], data['price'], color='blue')

area_range = np.linspace(data['area'].min(), data['area'].max(), 100)
bedrooms_range = np.linspace(data['bedrooms'].min(), data['bedrooms'].max(), 100)
area_grid, bedrooms_grid = np.meshgrid(area_range, bedrooms_range)

price_grid = model.predict(np.c_[area_grid.ravel(), bedrooms_grid.ravel()]).reshape(area_grid.shape)

ax.plot_surface(area_grid, bedrooms_grid, price_grid, color='red', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.legend()
plt.show()