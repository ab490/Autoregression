#Name: Anooshka Bajaj


import pandas as pd
df=pd.read_csv(r'E:\datasetA6_HP.csv')
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR

#1

#1(a)
plt.plot(range(len(df)),df['HP'])
plt.xlabel('Date')
plt.ylabel('Power Consumed (MW)')
plt.show()

#1(b)
t=df['HP']
t_1=df['HP'][1:]                   #one day lag time sequence
print('\nAutocorrelation :',np.corrcoef(t_1,t[:-1])[1][0])

#1(c)
plt.scatter(t[:-1],t_1, marker='*')
plt.xlabel('Given Time Sequence')
plt.ylabel('One-Day Lagged Generated Sequence')
plt.title('Scatter Plot')
plt.show()

#1(d)
Autocorrelation=[]
for p in range(1,8):
    t_p=df['HP'][p:]
    Autocorrelation.append(np.corrcoef(t_p,t[:-p])[1][0])
    
plt.scatter(range(1,8),Autocorrelation)
plt.plot(range(1,8),Autocorrelation)
plt.xlabel('Lagged Value')
plt.ylabel('Correlation Coefficient')
plt.title('Autocorrelation')
plt.show()

#1(e)
sm.graphics.tsa.plot_acf(df['HP'],lags=range(1,8))
plt.xlabel('Lagged Value')
plt.ylabel('Correlation Coefficient')
plt.title('Autocorrelation')
plt.show()

#2
train = t[1:len(t)-250]              #first 250 days as train data
test = t[len(t)-250:]                #last 250 days as test data
test_t=test.values[:-1]
test_t_1=test.values[1:]

print('\nRMSE of Persistance Model:')
print((((test_t)-(test_t_1))**2).mean()**0.5)

#3

#3(a)
model=AR(train)                     #training AR model
model_fit = model.fit(5)
                                    #predicting the values for the test dataset
prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse=((prediction-test)**2).mean()**0.5
print('\nRMSE of AR(5) Model:', rmse)

plt.scatter(test,prediction, marker='*')
plt.xlabel('Original Test Data')
plt.ylabel('Predicted Test Data')
plt.title('AR(5)')
plt.show()

#3(b)
Lag=[1,5,10,15,25]
RMSE=[]
for k in Lag:
    model=AR(train)
    model_fit = model.fit(k)
    prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    rmse=((prediction-test)**2).mean()**0.5
    RMSE.append(rmse)
    print('\nRMSE for lag',k,':\n',rmse)

#3(c)
x_t=train

for p in range(1,len(train)):
    x_t_p=train[p:]
    if abs(np.corrcoef(x_t_p,x_t[:-p])[1][0]) < 2/len(train)**0.5:
        p=p-1
        print('Heuristic Value for Optimal Number of Lags :',p);break

model=AR(train.values)
model_fit = model.fit(p)
prediction=model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse=((prediction-test)**2).mean()**0.5
print('\nRMSE for Optimal Lags:',rmse)


#3(d)

print('\nWithout using Heuristics for Calculating Optimal Lag:')
print('\np:')
optimal_index=RMSE.index(min(RMSE))
print(Lag[optimal_index])
print('\nRMSE:')
print(RMSE[optimal_index])

print('\nUsing Heuristics for Calculating Optimal Lag:')
print('\np:')
print(p)
print('\nRMSE:')
print(rmse)




