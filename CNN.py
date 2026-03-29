# CNN TensorFlow Execution using MNIST Dataset

"""
- **CNN TensorFlow execution using MNIST Dataset**
- **The given code is a Convolutional Neural Network (CNN) for Handwritten Digit Classification**
- It teaches a computer to recognize digits (0–9) from images using spatial feature extraction
- Dataset used is MNIST Dataset
- MNIST = 70,000 handwritten digits (28×28 grayscale images)
- CNN is more powerful than a simple neural network because it preserves image structure
"""
#========================================================================================================
# Load Libraries 
#========================================================================================================
import numpy as np                                      # for numerical operations
import matplotlib.pyplot as plt                         # for visualization
from tensorflow.keras.datasets import mnist             # dataset
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix            # evaluation
#========================================================================================================
"""Load Dataset
# Explanation:
# - Loads MNIST dataset
# - Dataset contains:
#   - 60,000 training images
#   - 10,000 test images
# - Each image:
#   - Size → 28 × 28 pixels
#   - Label → digit (0–9)
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#========================================================================================================
# Data Processing
"""
Data preprocessing is crucial before feeding into CNN:

1. Normalization
   - Scale pixel values between 0–1
   - Correct value should be /255 (not /225)

2. Reshaping
   - CNN requires 4D input:
     (samples, height, width, channels)
   - Here:
     (60000, 28, 28, 1)
     (10000, 28, 28, 1)
"""
train_images = train_images.astype('float32')/225
test_images = test_images.astype('float32')/225

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
#========================================================================================================
# Building the CNN Model
"""
Architecture of CNN:

1. Conv2D Layer
   - 32 filters, size (3×3)
   - Activation: ReLU
   - Extracts basic features (edges, patterns)

2. MaxPooling Layer
   - Reduces image size
   - Keeps important features

3. Conv2D Layer
   - 64 filters
   - Learns more complex patterns

4. MaxPooling Layer
   - Further downsampling

5. Flatten Layer
   - Converts 2D feature maps → 1D vector

6. Dense Layer
   - 64 neurons (learning complex relationships)

7. Output Layer
   - 10 neurons (digits 0–9)
   - Softmax → probability distribution
"""
model = Sequential ([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
        ])
#========================================================================================================
# Compile the Model
"""
- Optimizer: Adam
   Automatically adjusts learning rate

- Loss Function:
   sparse_categorical_crossentropy
   Used when labels are integers (0–9)

- Metric:
   Accuracy
"""

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
#========================================================================================================
# Train the Model
"""
What happens:
- Model learns spatial features from images
- Runs 5 epochs over dataset
"""
model.fit(train_images, train_labels, epochs = 5)
#========================================================================================================
# Evaluate the Model
"""
Purpose:
- Test model on unseen data

Outputs:
- Loss
- Accuracy (CNN usually performs better than simple NN)
"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy:{test_acc}")
#========================================================================================================
# Save the Model
"""
Explanation:
- Saves trained CNN model
- Can reuse without retraining
"""
model.save("mnist_cnn.h5")
print("Model Saved Successfully")
#========================================================================================================
# Predictions
"""
Explanation:
- Model predicts probability for each digit
- argmax() selects highest probability → predicted digit
"""
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

print("First 10 Predictions")
print(predicted_labels[:10])

print("First Actual 10 Predictions")
print(test_labels[:10])
#========================================================================================================
# Confusion Matrix
"""
Purpose:
- Shows correct vs incorrect predictions
- Helps analyze model performance
"""
cm = confusion_matrix(test_labels, predicted_labels)
print(f"Confusion Matrix of the model is")
print(cm)
#========================================================================================================
# Visualization
"""
What it does:
- Displays first 5 test images
- Shows:
   Predicted label
   Actual label
"""
for i in range(5):
    plt.imshow(test_images[i].reshape(28,28), cmap = 'grey')
    plt.title(f"Predicted:{predicted_labels[i]} | Actual :{test_labels[i]}")
    plt.axis('off')
    plt.show()
#========================================================================================================
"""
Summary:
- This project builds a Convolutional Neural Network (CNN) using TensorFlow
- CNN captures spatial features better than simple neural networks
- Achieves high accuracy in handwritten digit classification
- Includes preprocessing, training, evaluation, prediction, confusion matrix, and visualization
"""