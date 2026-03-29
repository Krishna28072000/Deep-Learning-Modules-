# Time Series Forecasting using GRU (Gated Recurrent Unit)

"""
- This project uses a GRU-based Recurrent Neural Network for time series forecasting
- It predicts the next day's temperature based on past temperature values
- GRU is a type of RNN that handles sequential data efficiently
- It captures temporal dependencies in data (time-based patterns)
"""
#========================================================================================================
#Load Libraries
#========================================================================================================
import numpy as np                                  # numerical operations
import pandas as pd                                 # data handling
from sklearn.preprocessing import MinMaxScaler      # normalization
from tensorflow.keras.models import Sequential      # model building
from tensorflow.keras.layers import Dense, GRU      # GRU + output layer
from tensorflow.keras.optimizers import Adam        # optimizer
#========================================================================================================
#Load Dataset
""" 
- Loads dataset from CSV file
- Dataset contains:
   Date column
   Temperature column
- Displays first 5 rows for inspection
"""
#========================================================================================================
df = pd.read_csv("D:\Practice\Tensorflow_and_keras_models\samples\data.csv")
print(f"THE FIRST 5 COLUMNS OF THE DATA IS :\n{df.head(5)}")
#========================================================================================================
#Data Preprocessing
""" 
Explanation:
1. Convert Date column → datetime format
2. Set Date as index (important for time series)
3. Select Temperature column only
4. Normalize data between 0–1 using MinMaxScaler
"""
#========================================================================================================
scaler = MinMaxScaler(feature_range=(0,1))
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Temperature']]  
scaled_data = scaler.fit_transform(data)
#========================================================================================================
#Create Dataset (Sliding Window)
""" 
Explanation:
- Converts time series into supervised learning format

Example:
Input (X) → last 100 days
Output (y) → next day temperature

- Uses sliding window technique
"""
#========================================================================================================

def create_dataset(data, time_step = 1):
    X, y = [], []
    for i in range(len(data)- time_step-1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
#========================================================================================================
#Prepare Input Data
""" 
Explanation:
- time_step = 100 → uses last 100 values to predict next
- Reshape required for GRU:
   (samples, time_steps, features)
"""
#========================================================================================================
time_step = 100
X,y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1],1)
#========================================================================================================
#Build GRU Model
""" 
Architecture:
1. GRU Layer (50 units)
   - return_sequences=True → passes sequence to next layer

2. GRU Layer (50 units)
   - Learns deeper temporal patterns

3. Dense Layer
   - Outputs single value (next temperature)
"""
#======================================================================================================== 
model = Sequential()
model.add(GRU(units = 50, return_sequences = True, input_shape = (X.shape[1], 1)))
model.add(GRU(units=50))
model.add(Dense(units = 1))
#========================================================================================================
#Compile the Model
""" 
Explanation:
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
   Suitable for regression problems
"""
#========================================================================================================
model.compile(optimizer = Adam(learning_rate =0.001), loss = 'mean_squared_error')
#========================================================================================================
#Train the Model
""" 
What happens:
- Model learns temperature patterns
- Uses batch training
- 10 epochs = 10 passes over dataset
"""
#========================================================================================================
model.fit(X,y, epochs = 10, batch_size = 32)
#========================================================================================================
#Make Prediction
""" 
Explanation:
- Takes last 100 values from dataset
- Used to predict next value
"""
#========================================================================================================
input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
print(f"INPUT SEQUENCE :{input_sequence}")
predicted_values = model.predict(input_sequence)
print(f"PREDICTED_SEQUENCES:{predicted_values}")
#========================================================================================================
#Inverse Scaling
""" 
Explanation:
- Converts normalized value back to original scale
- Gives actual temperature prediction
"""
#========================================================================================================

predicted_values = scaler.inverse_transform(predicted_values)
print(f"The predicted temperature for the next day is: {predicted_values[0][0]:.2f}°C")
#========================================================================================================
#Example Output
#========================================================================================================
"""
PREDICTED_SEQUENCES:[[0.4669627]]
# The predicted temperature for the next day is: 24.34°C
"""
#========================================================================================================
#Summary
#========================================================================================================
"""
- This project uses a GRU-based RNN for time series forecasting
- Predicts next day's temperature using past data
- Uses sliding window technique for sequence creation
- Applies normalization for better model performance
- Outputs real-world temperature prediction after inverse scaling
"""