"""
Display mnist data for 5 digit
Date: 2025-02-02
Author: Holmes Amzish
"""

import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

images = train_images[:5]
labels = train_labels[:5]

plt.figure(figsize=(6, 2))

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray_r')
    plt.title(f'Label: {labels[i]}')
    plt.axis('off')

plt.show()
