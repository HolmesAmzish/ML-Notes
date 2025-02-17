import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def read_points(filename):
    points = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            x1, x2, tag = line.strip().split(',')
            points.append([float(x1), float(x2)])
            labels.append(int(tag))
    return np.array(points), np.array(labels)


points, labels = read_points('points.csv')
train_points = points[:200]
train_labels = labels[:200]


# 绘图函数，将标签映射为颜色
def data_to_color(tags):
    return ['red' if tag == 1 else 'blue' for tag in tags]


# 绘制训练数据
plt.figure(figsize=(12, 8))
plt.scatter(
    train_points[:, 0], train_points[:, 1],
    c=data_to_color(train_labels),
    alpha=0.5
)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Original Points Data')
plt.show()

# 创建神经网络模型
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(2,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 训练模型
test_points = points[200:]
test_labels = labels[200:]

model.fit(train_points, train_labels, epochs=5, batch_size=32)  # 使用合适的批处理大小

# 预测
predictions = model.predict(test_points)
prediction_labels = predictions.flatten()

print("Predicted labels:", prediction_labels)
