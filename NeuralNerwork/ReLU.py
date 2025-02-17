import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


x = np.linspace(-10, 10, 400)
y = relu(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ReLU', color='blue')
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.grid(True)
plt.show()