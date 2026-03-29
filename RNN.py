# Character-Level Text Generation using SimpleRNN

"""
- **RNN TensorFlow execution for Text Generation**
- The given code builds a Character-Level Language Model
- It learns patterns from text and generates new text character by character
- Uses SimpleRNN (Recurrent Neural Network)
- RNN is suitable for sequential data like text
- The model predicts the next character based on previous characters
"""
#========================================================================================================
#Load Libraries
#========================================================================================================

import numpy as np                                      # numerical operations
import tensorflow as tf                                 # deep learning framework
from tensorflow.keras.models import Sequential          # model building
from tensorflow.keras.layers import SimpleRNN, Dense    # RNN + output layer

#========================================================================================================
#Input Text Data
#========================================================================================================
"""
Explanation:
- This is the training data
- The model learns patterns from this text
- Repeating text increases dataset size
"""

text = """
This is GeeksforGeeks is a computer science portal for geeks.
It contains well written tutorials, programming problems,
interview experiences and coding practice problems.
"""

text = text * 50
#========================================================================================================
#Character Mapping
#========================================================================================================
""" 
- Extract unique characters
- Create mappings:

char_to_index:
   'a' → 0
   'b' → 1
   ...

index_to_char:
   0 → 'a'
   1 → 'b'

- Helps convert text → numbers (required for model)
"""

chars = sorted(list(set(text)))
char_to_index = {char : i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
#========================================================================================================
#Sequence Preparation
#========================================================================================================
""" 
- Input: 40 characters
- Output: next character

Example:
Input  → "This is GeeksforGeeks is a compu"
Output → 't'

- Sliding window approach
"""
seq_length = 40
sequences = []
labels = []
for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])
#========================================================================================================
#Convert to Arrays
#========================================================================================================
x = np.array(sequences)
y = np.array(labels)
#========================================================================================================
#One-Hot Encoding
#========================================================================================================
""" 
- Converts integers → vectors
Example:
'a' → [1,0,0,0,...]
'b' → [0,1,0,0,...]

- Required for neural networks
"""
X_one_hot = tf.one_hot(x, len(chars))
Y_one_hot = tf.one_hot(y, len(chars))
#========================================================================================================
#Building the RNN Model
#========================================================================================================
""" 
1. SimpleRNN Layer:
   - 150 neurons
   - Learns sequential dependencies

2. Dense Layer:
   - Output size = number of characters
   - Softmax → probability of next character
"""
model = Sequential()
model.add(SimpleRNN(150, input_shape = (seq_length, len(chars)), activation = "tanh"))
model.add(Dense(len(chars), activation = "softmax"))
#========================================================================================================
#Compile the Model
#========================================================================================================
""" 
- Optimizer: Adam
- Loss: categorical_crossentropy (multi-class classification)
- Metric: Accuracy
"""
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
#========================================================================================================
#Train the Model
#========================================================================================================
""" 
- Model learns character patterns
- Higher epochs → better text generation
- 300 epochs = deep learning of text structure
"""
model.fit(X_one_hot,Y_one_hot, epochs = 300)
#========================================================================================================
#Text Generation
#========================================================================================================
""" 
Steps:
1. Take last 40 characters
2. Predict next character
3. Append it to text
4. Repeat

Two methods:
- argmax → most likely character (deterministic)
- random choice → more creative output
"""
start_seq = "This is GeeksforGeeks is a computer scie"
generated_text = start_seq
for i in range(50):
    x = np.array([[char_to_index[char] for char in generated_text[-seq_length:]]])
    X_one_hot = tf.one_hot(x,len(chars))
    prediction = model.predict(X_one_hot, verbose=0)[0]
    # next_index = np.random.choice(len(chars), p=prediction)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    generated_text += next_char
#========================================================================================================
#"Output Generated Text
#=======================================================================================================
# The generated text is a continuation of the input sequence, created by the model based on learned patterns. It may not be perfect but demonstrates the model's ability to generate text character by character.
# Note: The output will vary each time you run the code due to the randomness in training and text generation (if using random choice).

print(f"The generated_text is :\n")
print(generated_text)
#========================================================================================================
#Summary
#========================================================================================================
"""
- This project builds a Character-Level RNN using TensorFlow
- It learns patterns from text and generates new text
- Uses SimpleRNN to handle sequential dependencies
- Includes preprocessing, encoding, training, and text generation
- Demonstrates basic Natural Language Processing (NLP)
"""