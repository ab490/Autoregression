# Autoregression
Autregression in python.

I am given the electricity power units (MegaWatts) consumed on a daily basis by Himachal Pradesh as a csv file. To study the power consumption in the light of COVID19, data is recorded in the form of a time series for a period of 17 months beginning from 2nd Jan 2019 till 23rd May 2020. Rows are indexed with dates, first column represents the date and second column represent power consumed in Himachal Pradesh. Rows and columns put together, each data point in second column reflects the power consumed in Mega Units (MU) by the Himachal Pradesh (column) at the given date (row).

1. Autocorrelation line plot with lagged values:\
a. Created a line plot with x-axis as index of the day and y-axis as power consumed in mega units (MU).\
b. Generated another time sequence with one day lag to the given time sequence. Found the Pearson correlation (autocorrelation) coefficient between the generated one day lag time sequence and the given time sequence.\
c. Generated a scatter plot between given time sequence and one day lagged generated sequence in 1.b. \
d. Generated multiple time sequences with different lag values (1 day, 2 days, 3 days upto 7 days). Computed the Pearson correlation (autocorrelation) coefficient between each of the generated time sequences and the given time sequence. Created a line plot between obtained correlation coefficients (on y-axis) and lagged values (on x-axis).\
e. Used python inbuilt function ‘plot_acf’ to generate the line plot which I manually coded in 1.d and observed the trend in line plot with increase in lagged values.

2. Considered the last 250 days as test data. Given a dataframe with data points for (t-1)th timestamp and (t)th timestamp, problem statement is to predict (t) th datapoint given (t-1)th datapoint. The persistence algorithm outputs the same value as input while the expected value is corresponding (t) th datapoint from the original dataframe. This is known as the persistence model and is the simplest autoregression model. Computed the RMSE between predicted power consumed for test data and original values for test data.

3. A general autoregression model estimates the unknown data values as a linear combination of given lagged data values. Did the following:\

   a. Split the data into two parts for training and testing. Chose the first 250 days as training data and last remaining days as test data. Generated an autoregression (AR)    model using AutoReg(). Used 5 lagged values as its input (p=5). Train/Fit the model onto the training dataset. Used the trained AR model to predict the values for the test    dataset. Computed RMSE computed for test data and compared it with RMSE obtained in part 2. Generated a plot between the original test data time sequence and predicted        test data time sequence. \

   b. Generated five AR models using AutoReg() function with lagged values as last 1, 5, 10, 15 and 25 days. Computed the RMSE between predicted and original data values and    infered the changes in RMSE with changes in lagged values. \

   c. Computed the heuristic value for optimal number of lags up to the condition on autocorrelation such that abs(AutoCorrelation) > 2/sqrt(T), where T is the number of 
   observations in training data. Used it as input in AutoReg()function to predict the power consumed in test days and compute the RMSE value. \

   d. Compared the optimal number of lags (p) in parts 3.b and 3.c. Compared the RMSE values in parts 3.b and 3.c.
