"""
Recognize mnist handwritten digits by Convolutional Neural Network model
Date: 2025-02-02
Author: Holmes Amzish
"""

from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

predictions = model.predict(test_images)
plt.imshow(test_images[0].reshape(28, 28), cmap='binary')
plt.show()
print(predictions[0])
print(predictions[0].argmax())
print(test_labels[0])