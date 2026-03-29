# LSTM for Time Series Forecasting
"""
- This project uses an LSTM-based Recurrent Neural Network for time series forecasting
- It predicts monthly milk production based on past production data
- LSTM (Long Short-Term Memory) is a type of RNN that captures long-term dependencies in sequential data
- It is effective for time series data where past values influence future values
- The model learns patterns in the data to make accurate predictions about future production"""
#========================================================================================================
#Load Libraries
import numpy as np                                  # numerical operations
import pandas as pd                                 # data handling 
import matplotlib.pyplot as plt                     # visualization
from sklearn.preprocessing import MinMaxScaler      # normalization
from sklearn.model_selection import train_test_split # for splitting data
from tensorflow.keras.models import Sequential      # model building
from tensorflow.keras.layers import Dense, LSTM, Dropout # LSTM + output layer
#========================================================================================================
#Load Dataset
"""
- Loads dataset from CSV file
- Dataset contains:
   Date column
   Production column (pounds per cow)
- Displays first 5 rows for inspection
"""
df = pd.read_csv("D:\Practice\Tensorflow_and_keras_models\monthly_milk_production.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
production = df['Production'].astype(float).values.reshape(-1,1)
#========================================================================================================
#Data Preprocessing
"""
Explanation:
1. Normalization
   - Scale production values between 0–1 using MinMaxScaler
2. Create Dataset
   - Input: last 12 months of production
   - Output: next month's production
   - Uses sliding window technique to create sequences for LSTM
3. Train-Test Split
    - 80% training data, 20% test data
    - shuffle=False to maintain time order
4. Reshape for LSTM
    - LSTM expects input shape: (samples, time_steps, features)
    - Here: (samples, 12, 1)
"""
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(production)

window_size = 12 
X = []
Y = []
target_dates = df.index[window_size:]

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i, 0])
    Y.append(scaled_data[i, 0])

X = np.array(X)
Y = np.array(Y)

X_train, x_test, Y_train, y_test, dates_train, dates_test = train_test_split(X, Y, target_dates, test_size = 0.2, shuffle= False)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
#========================================================================================================
#Build LSTM Model
"""
- LSTM Layer 1: 128 units, returns sequences for next LSTM layer
- Dropout Layer 1: 20% dropout to prevent overfitting
- LSTM Layer 2: 128 units, returns final output
- Dropout Layer 2: 20% dropout
- Dense Output Layer: 1 unit for predicting production
- Compile with Adam optimizer and mean squared error loss
- Train for 100 epochs with batch size of 32, using 20% of training data for validation
"""
model = Sequential()
model.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 128))
model.add(Dropout(0.2))
model.add(Dense(1))
#========================================================================================================
#Compile and Train the Model
"""
- Optimizer: Adam (adaptive learning rate)
- Loss: mean_squared_error (regression problem)
- Metrics: accuracy (not ideal for regression but included for monitoring)
- Training: 100 epochs, batch size 32, validation split 0.2
- Model learns patterns in the data to predict future production"""
model.compile(optimizer = "adam", loss = "mean_squared_error")
history = model.fit(X_train,Y_train, epochs = 100, batch_size = 32,validation_split = 0.2)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
#========================================================================================================
#Evaluate the Model
"""
- Calculate Root Mean Squared Error (RMSE) to evaluate prediction accuracy
- RMSE gives an idea of how far predictions are from actual values
- Plot actual vs predicted production over time to visually assess model performance
- A good model will have predictions closely following the actual production trend
"""

rmse = np.sqrt(np.mean((y_test - predictions)**2))
print(f'RMSE: {rmse:.2f}')
#========================================================================================================
#Plot Actual vs Predicted Production
"""
- Plots actual production (y_test) and predicted production (predictions) over time
- X-axis: Date
- Y-axis: Production (pounds per cow)
- Helps visualize how well the model captures trends and patterns in the data
- A good fit will show the predicted line closely following the actual line
"""
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Production')
plt.plot(dates_test, predictions, label='Predicted Production')
plt.title('Actual vs Predicted Milk Production')
plt.xlabel('Date')
plt.ylabel('Production (pounds per cow)')
plt.legend()
plt.show()
#========================================================================================================
#Summary
"""
- This code demonstrates how to use an LSTM-based RNN for time series forecasting
- It includes data preprocessing, model building, training, and evaluation
- The model learns from past production data to predict future production, with performance evaluated using RMSE and visualized through plots
- LSTM is effective for capturing temporal dependencies in time series data, making it a powerful tool for forecasting tasks like this one
"""