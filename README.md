This code generates ML predictions for historical stock prices using Keras, NumPy, and Pandas.


Process:
    Data Loading: Reads a CSV file containing historical stock prices (e.g., Disney).
    Data Scaling: Uses MinMaxScaler to scale the data for LSTM processing.
    Data Splitting: Divides data into 80% training and 20% testing (pandemic period).
    Sequence Creation: create_sequences function generates input and output sequences with a sequence length of 10.
    Model Building and Training: Builds an LSTM model, trained over 10 epochs with a batch size of 32.
    Evaluation: Uses the Adam optimizer and mean squared error to evaluate prediction accuracy.
    Forecasting: Uses the model to predict stock prices, reverting values to their original scale and printing the results.
