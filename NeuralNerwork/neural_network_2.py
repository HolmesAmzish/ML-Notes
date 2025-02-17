import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from file
data = pd.read_csv('points.csv', header=None, names=['x1', 'x2', 'label'])
X = data[['x1', 'x2']].values
y = data['label'].values

# Split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

color = {
    0: 'blue',
    1: 'red'
}
plt.figure(figsize=(12, 8))
colors = [color[label] for label in y_train]
plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=colors, alpha=0.5, label=colors)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Train data')
plt.show()



# Create the neural network
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(2,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=4)
print("Training complete.\n")

loss, accuracy = model.evaluate(X_test, y_test)
y_hat_pred = model.predict(X_test)
y_hat = (y_hat_pred > 0.5).astype(int).flatten()

shape = {
    True: 'o',
    False: 'x'
}

plt.figure(figsize=(12, 8))
colors = [color[label] for label in y_test]
shapes = [shape[y_ == y_hat_] for y_, y_hat_ in zip(y_test, y_hat)]

for x1, x2, c, s in zip(X_test[:, 0], X_test[:, 1], colors, shapes):
    plt.scatter(x1, x2, color=c, marker=s, alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Test data')
plt.show()

print(f'Test Accuracy: {accuracy:.4f}')