"""
Autoencoder on MNIST Dataset
----------------------------
This script builds and trains a simple Autoencoder using TensorFlow/Keras
to reconstruct handwritten digit images from the MNIST dataset.

It also visualizes the original and reconstructed images in a single figure.

Author: Your Name
"""
#-----------------------------------------
# Import Libraries #
#-----------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.datasets import mnist


# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------
"""
- Load MNIST dataset (28x28 grayscale images)
- Normalize pixel values to range [0, 1]
- Reshape data to include channel dimension (required for CNN/NN input)
"""
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Add channel dimension → (28, 28, 1)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# -----------------------------
# 2. Define Autoencoder Model
# -----------------------------
class AutoEncoder(Model):
    """
    Autoencoder Model:
    - Encoder: Compresses input image into a lower-dimensional latent vector
    - Decoder: Reconstructs image from latent representation
    """

    def __init__(self, latent_dimension):
        super(AutoEncoder, self).__init__()

        # Encoder: Reduces dimensionality
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(latent_dimension, activation='relu'),
        ])

        # Decoder: Reconstructs original image
        self.decoder = tf.keras.Sequential([
            layers.Dense(28 * 28, activation="sigmoid"),
            layers.Reshape((28, 28, 1))
        ])

    def call(self, input_data):
        """
        Forward pass:
        - Encode input → latent space
        - Decode latent vector → reconstructed image
        """
        encoded = self.encoder(input_data)   # ⚠️ Fixed: removed tuple bug
        decoded = self.decoder(encoded)
        return decoded


# -----------------------------
# 3. Compile and Train Model
# -----------------------------
"""
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Train model to reconstruct input images
"""
latent_dimension = 64
autoencoder = AutoEncoder(latent_dimension)

autoencoder.compile(
    optimizer='adam',
    loss=losses.MeanSquaredError()
)

autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)


# -----------------------------
# 4. Generate Reconstructions
# -----------------------------
"""
- Encode test images
- Decode them back to reconstructed images
"""
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# -----------------------------
# 5. Visualization
# -----------------------------
"""
Display original and reconstructed images in a single figure:
- Top row: Original images
- Bottom row: Reconstructed images
"""
n = 6  # Number of images to display

plt.figure(figsize=(12, 4))

for i in range(n):
    # Original images (top row)
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed images (bottom row)
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()


# -----------------------------
# 6. Summary
# -----------------------------
"""
This Autoencoder learns to compress and reconstruct handwritten digits.

Key Concepts:
- Encoder → Feature extraction / compression
- Decoder → Reconstruction
- Latent Space → Compressed representation of input

Use Cases:
- Image denoising
- Dimensionality reduction
- Anomaly detection
- Feature learning
"""