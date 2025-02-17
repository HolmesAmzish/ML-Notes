"""
Recognize handwrite digit of mnist data set by classic neural network
Date: 2025-02-02 20:32
Author: Holmes Amzish
"""

from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

predictions = model.predict(test_images)
plt.imshow(test_images[0], cmap='binary')
plt.show()
print(predictions[0])
print(predictions[0].argmax()) # Get prediction
print(test_labels[0]) # Get label