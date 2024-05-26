import numpy as np #fundamental package for scientific computing
import pandas as pd #data structures to work with labeled data
from sklearn.preprocessing import MinMaxScaler #scales the data
from tensorflow import keras #keras is a neural network library
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Loading the historical stock prices
df = pd.read_csv('./DIS.csv') 

# Scaling the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Splitting the data into training and testing sets
train_size = int(len(scaled_data) * 0.8) #80% of the data file is for training, 20% for testing
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create input and output sequences
def create_sequences(data, sequence_length):
  X = []
  y = []
  for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])
  return np.array(X), np.array(y)
  
# Set the sequence length
sequence_length = 10 #In create_sequences, takes the previous 10 closing prices as input
# Create input and output sequences for training
X_train, y_train = create_sequences(train_data, sequence_length)
# Create input and output sequences for testing
X_test, y_test = create_sequences(test_data, sequence_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32) #epochs -> the number of 
times the dataset is passed through

# Evaluate the model on the testing data
loss = model.evaluate(X_test, y_test)
print('Testing Loss:', loss)

# Make predictions
predictions = model.predict(X_test)
# Inverse scaling of the predictions to revert them back to their original scale
predictions = scaler.inverse_transform(predictions)
# Print the predicted prices
for i in range(len(predictions)):
  print(predictions[i][0])
