# Importing the libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

import urllib.request
import pickle
from geopy.geocoders import Nominatim
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

pd.set_option('display.max_columns', None)



# def predict(start_date,end_date,address):
#   # start_date = input("Enter start date in yyyy/mm/dd format:")
#   start_date = start_date

#   start_date = start_date.replace('/','')
#   # end_date = input("Enter end date in yyyy/mm/dd format:")
#   end_date = end_date

#   end_date = end_date.replace('/','')
#   # address = input("Enter City Name:")
#   address = address


#   geolocator = Nominatim(user_agent="Chaitanya")
#   location = geolocator.geocode(address)
#   print(location.address)
#   print((location.latitude, location.longitude))
#   latitude = location.latitude
#   longitude = location.longitude

#   url = 'https://power.larc.nasa.gov//api/temporal/daily/point?parameters=T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,PS,WS10M_RANGE&community=SB&longitude={}6&latitude={}&start={}&end={}&format=CSV'.format(longitude,latitude,start_date,end_date)
#   urllib.request.urlretrieve(url,'weather.csv')
  
# predict("2024/04/14","2020/04/14","Nagpur")
# predict()
  
df = pd.read_csv('Nagpur_Daily_20140415_20240415.csv',skiprows=14,encoding='utf-8')
  
df['YEAR'] = df.YEAR.astype(str)
df['MO'] = df.MO.astype(str)
df['DY'] = df.DY.astype(str)

df['date'] = df['YEAR'].str.cat(df['MO'], sep = '/')
df['DATE'] = df['date'].str.cat(df['DY'], sep = '/')
# df.head()

# removing unrequired attributes
df.drop(columns=['YEAR','MO','DY','date'],axis=1,inplace=True)

df.set_index(['DATE'], inplace = True)

# print(df.head())
# print(df.dtypes)
# print(df.info())

# with pd.option_context('display.max_rows', None,
#                        'display.max_columns', None,
#                        'display.precision', 3,
#                        ):
#     print("")

# print(df_scaled)


# Scaling the data - Normalize (0-1) or Standardize (gaussian data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
# print('Scaled df:\n', df_scaled, '\n', df_scaled.shape)

# Splitting the dataset

# Train - Test
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_scaled, test_size = 0.2, shuffle = False)

# X - Y
x_train, y_train, x_test, y_test = [], [], [], []
for i in range (1, len(train)):
    x_train.append(train[i-1])
    y_train.append(train[i])
for i in range (1, len(test)):
    x_test.append(test[i-1])
    y_test.append(test[i])

pd.DataFrame(x_train)
pd.DataFrame(y_train)


# Converting list to array
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


#define model 
model = Sequential()
model.add(Dense(6, input_dim = 6, activation = 'relu')) # input neaurons counts
model.add(Dense(8, activation = 'relu'))  # hidden layer neaurons counts
model.add(Dense(8, activation='relu'))
model.add(Dense(6)) # output neaurons counts

model.summary()

#compile  model
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])


# history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 200, batch_size = 15, shuffle = False)
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 200, batch_size = 15, shuffle = False)



# y_pred = model.predict(x_test)
# y_pred = scaler.inverse_transform(y_pred)
# actual_y_pred = scaler.inverse_transform(y_test)


# plt.rcParams["figure.figsize"] = (10,6)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Training and Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# print('Actual values:')
# print(' T2M_MAX  T2M_MIN RH2M  PRECTOTCORR PS  WS10M_RANGE')
# print(pd.DataFrame(actual_y_pred))

# print()
# print('-----------------------------------------------------------------------------------')
# print()
# print('Predicted values:')
# print('       T2M_MAX    T2M_MIN    RH2M    PRECTOTCORR   PS    WS10M_RANGE')
# print(pd.DataFrame(y_pred))


# Evaluating the model
# scores = model.evaluate(actual_y_pred, y_pred, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# T = range(y_pred.shape[0])


# from sklearn.model_selection import KFold
# cv = KFold(n_splits=10, shuffle=False)

# # Iterate through CV splits
# results = []
# for tr, tt in cv.split(x_train, y_train):
#     # Fit the model on training data
#     model.fit(x_train[tr], y_train[tr])

#     # Generate predictions on the test data and collect
#     prediction = model.predict(x_train[tt])
#     results.append((prediction, tt))
    

# input_data = []

# pickle.dump(cv,open("weather.pkl","wb"))

# print('Enter the weather parameters of previous day: ')

# attr1 = float(input("Enter Maximum Temperature: "))
# attr2 = float(input("Enter Minimum Temperature: "))
# attr3 = float(input("Enter Relative Humidity:"))
# attr4 = float(input("Enter Precipitation: "))
# attr5 = float(input("Enter Surface Pressure: "))
# attr6 = float(input("Enter Wind Speed at 10M Range in km/h : "))

# attr1 = float(input("Enter Maximum Temperature: "))
# attr2 = float(input("Enter Minimum Temperature: "))
# attr3 = float(input("Enter Relative Humidity:"))
# attr4 = float(input("Enter Precipitation: "))
# attr5 = float(input("Enter Surface Pressure: "))
# attr6 = float(input("Enter Wind Speed at 10M Range in km/h : "))

# input_data.append(attr1)
# input_data.append(attr2)
# input_data.append(attr3)
# input_data.append(attr4)
# input_data.append(attr5)
# input_data.append(attr6)

# input_data = np.array(input_data)
# input_data.shape = (1,6)

# print()
# print('---------------------------------------------------------------------------------------')
# print('Input Data: ', input_data)
# input_data = scaler.transform(input_data)
# print('Scaled Input Data:', input_data)

# print()
# pred1 = model.predict(input_data)
# pred2 = scaler.inverse_transform(pred1)

# print('Predicted Data: ')
# print(pd.DataFrame(pred2))

# print()
# print('--------------------------------------------------------------------------------------')
# print()
# print('Predicted Values:')
# print('Maximum Temperature:', pred2[0][0])
# print('Minimum Temperature:', pred2[0][1])
# print('Relative Humidity:', pred2[0][2])
# print('Precipitation:', pred2[0][3])
# print('Surface Pressure:', pred2[0][4])
# print('Wind Speed at 10m range:', pred2[0][5])