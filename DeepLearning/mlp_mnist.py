from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the input image to a vector of size 784
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images = train_images.astype("float32") / 255
# test_images = test_images.astype("float32") / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128)

predictions = model.predict(test_images)
plt.imshow(test_images[0], cmap='binary')
plt.show()
print(predictions[0])
print(predictions[0].argmax())
print(test_labels[0])