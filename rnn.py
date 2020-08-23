# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# The column we want is "Open" so we take its index from 1 to 2.
# This way we have a numpy array of 1 column.
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# Two ways of applying feature scaling - standardization & normalization.
from sklearn.preprocessing import MinMaxScaler

# sc is object of MinMaxScaler class.
# Feature range is between 0 and 1, as we apply normalization and 
# all stock prices will be in that range.
sc = MinMaxScaler(feature_range = (0, 1))

# Apply normalization on the training set.
# fit_transform - a method of MinMaxScaler class
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output so to make sure what the RNN remembers.
# 60 timesteps - are the past information from which our RNN 
# learns some trends and based on theses predicts the next values.
X_train = []
y_train = []

for i in range(60, 1258):
    
    # X_train will have 60 previous stock prices.
    X_train.append(training_set_scaled[i-60:i, 0])
    # Y_train will have the next day's stock price.
    y_train.append(training_set_scaled[i, 0])
    
    
# Convert lists to arrays, as RNN onloy accept arrays as arguments.
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping - adding more dimensionality to the data structure, so make x_train 3D which is 2D right now.
# X_train.shape[0] - This will give number of rows - nos of stock prices - This is the first dimension
# X_train.shape[1] - This will give number of columns - here will be number of time_steps - This is the second dimension
# 1 -> the number of indicators/predictors - This is the third dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
# Import sequential as this will allow us to build a neural network of layers.
from keras.models import Sequential
# Import dense for the output layer
from keras.layers import Dense
# Import LSTM for the LSTM layer
from keras.layers import LSTM
# Import dropout to add dropout regularization.
from keras.layers import Dropout

# Initialising the RNN
# Named regressor as we are predicting some continuous value, therefore doing some regression.
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# units = 50, To include large number of neurons for complex calculations - better results to capture upward and downward trend.
# We build a stacked LSTM, which will have several LSTM layers therefore we set return_sequences = True
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# We apply Dropout regularisation to avoid any overfitting.
# Specify number of neurons we want to ignore during regularization, classic is 20% so add 0.2
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
# We dont have return_sequences = True, as there are no LSTM layers after this
# so we want it be false, which is anyways the default value.
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()