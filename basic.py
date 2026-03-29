#Basic Tensorflow Execution using MNIST Dataset

"""
- **Basic Tensorflow execution using MNIST Dataset**
- **The given code is Simple Handwritten Digit Classifier**
- It generally teaches a computer to recognize digits (0-9) from images 
- Dataset used is MNIST Dataset
- MNIST = 70,000 handwritten digts these are generally 2D shaped
- Let's start with execution of the model, the model is a Simple Neural Network which generally a connections of neurons and weights associated to them
"""


#========================================================================================================
# Load Libraries 
#========================================================================================================
import numpy as np                                      # for numerical operations
import matplotlib.pyplot as plt                         # for data visualization
from sklearn.metrics import confusion_matrix            # for evaluating model performance
from tensorflow.keras.datasets import mnist             # for loading the MNIST dataset
from tensorflow.keras.layers import Dense               # for building the neural network layers
from tensorflow.keras.models import Sequential          # for creating the neural network model
import tensorflow as tf                                 # for using TensorFlow functionalities

#========================================================================================================
"""Load Dataset
# Explanation:
# - Loads MNIST dataset
# - Dataset contains:
# - 60,000 training images
# - 10,000 test images
# - Each image is:
# - Size → 28 × 28 pixels
# - Label → digit 0–9
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#========================================================================================================
# Data Processing :  The data is getting preprocessed before feeding it to the model, this is a crucial step in machine learning as it can significantly impact the performance of the model. The data processing steps include:
"""1.Reshaping
    # Converts image from:
    - 28 × 28 → 784 (1D vector)
    - Because neural networks take flat input

    2. Normalization
    - Dividing pixel values to scale between 0-1
    - Correct should be: /255
"""

train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32')/255
test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32')/255

#========================================================================================================
# Building the Model: The model is a simple feedforward neural network with the following architecture:
"""
- Layer 1 (Hidden Layer)
    128 neurons
    Activation: ReLU
    Input: 784 features
- Layer 2 (Output Layer)
    10 neurons (digits 0-9)
    Activation: Softmax
    Gives probability for each digit
"""

model = Sequential ([
Dense(128, activation='relu', input_shape=(28 * 28,)),
    Dense(10, activation = 'softmax')
])

#========================================================================================================
# Compile the Model : Compiling the model involves specifying the optimizer, loss function, and evaluation metrics. In this case:
"""
- Optimizer: Adam
    Automatically adjusts learning rate
- Loss Function:
    sparse_categorical_crossentropy
    Used when labels are integers (0–9)
- Metric:
    Accuracy
"""

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#========================================================================================================
# Train the Model : The model is trained using the fit method, which takes the training data and labels, and the number of epochs (iterations) to train for. In this case, the model is trained for 5 epochs.
"""
What happens:
- Model learns patterns from data
- Runs 5 epochs (full dataset passes)
"""

model.fit(train_images, train_labels,epochs = 5)

#========================================================================================================
# Evaluate the Accuracy : After training, the model's performance is evaluated on the test dataset using the evaluate method. This method returns the loss and accuracy of the model on the test data.
"""
Purpose:
- Tests model on unseen data
Outputs:
- Loss
- Accuracy (usually ~97-98%)

"""

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy", test_acc)

#========================================================================================================
# Saving the Model : The trained model is saved to a file named "mnist_model.h5" using the save method. This allows you to reuse the model later without having to retrain it.
""" 
Explanation:
- Saves trained model
- Can reuse later without retraining
"""
model.save("mnist_model.h5")
print("Model Saved successfully")

#========================================================================================================
# Manually Performing the predictions : The model is used to make predictions on the test dataset using the predict method. The predicted labels are obtained by taking the argmax of the predictions, which gives the index of the highest probability (the predicted digit).
"""
Explanation:
- Model outputs probabilities like: [0.1, 0.05, 0.8, ...]
- argmax() picks highest probability → predicted digit
"""
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis = 1)
print("\nFirst 10 predictions:")
print(predicted_labels[:10])

print("\n Actual Labels:")
print(test_labels[:10])

#========================================================================================================
# Creating Confusion Matrix : A confusion matrix is created using the confusion_matrix function from sklearn.metrics. This matrix shows the number of correct and incorrect predictions for each class, providing insight into the model's performance.
"""
Purpose:
It Shows:
- Correct predictions
- Misclassifications

Example:
- Actual vs Predicted
"""
cm = confusion_matrix(test_labels, predicted_labels)
print("\n Confusion Matrix : ")
print(cm)

#========================================================================================================
# Visual representation of Data Matches : Finally, the code visualizes the first 5 test images along with their predicted and actual labels using matplotlib. This helps to visually confirm the model's predictions.
"""
What it does:
- Displays first 5 images
Shows:
- Predicted label
- Actual label
"""
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels[i]} | Actual: {test_labels[i]}")
    plt.axis('off')
    plt.show()
#========================================================================================================
"""Summary : 
- This project builds a simple neural network using TensorFlow to classify handwritten digits from the MNIST dataset. 
- The model consists of a fully connected architecture with one hidden layer and achieves high accuracy on test data. 
- It includes preprocessing, training, evaluation, prediction, confusion matrix generation, and visualization.
"""