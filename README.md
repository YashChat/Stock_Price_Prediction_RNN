# Stock_Price_Prediction_RNN
We implement a Recurrent Neural Network with multiple LSTM(Long Short Term Memory) layers, and use this to predict the stock. 
In this particular example, I am using a dataset with google price details, and so it will predict the google stock price. 
But the dataset, can be modified easily to predict any stock.

Some important notes -
Two ways of applying feature scaling – 
•	Normalization – Whenever we build RNN, also if the sigmoid function is the activation function, then this is recommended 
ex – Prediction of stock prices
•	Standardization
 
All our scaled stock prices will be between 0 and 1 after normalization.
