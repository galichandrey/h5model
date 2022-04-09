# IMPORTING IMPORTANT LIBRARIES
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import datetime

import numpy as np 


# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY

# FOR REPRODUCIBILITY
np.random.seed(4)

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
# Binance dependencies that we are importing here !

apikey = 'Nfsrvy1PGQkwqQfefM5fgR1JjdHyQr2CWZhKSjTMSTtopbg0vUoLjTSvFTI4D4Mo'
secret = 'e5tDmfVaMROqTLD17wNChVUD6WpesIeas9vdgrybPL33NDVCDP3lQvTvP7GmKiqu'
client = Client(apikey, secret)
#set-up th client and authenticate the binance

crypto = client.get_all_tickers()
#crypto
# Getting every singly currency pair with respect to their tickers & price (USDD)

#crypto[1]['price']

crypto_df = pd.DataFrame(crypto)
#Converting these strings into Dataframe

crypto_df.set_index('symbol', inplace=True)

float(crypto_df.loc['BTCUSDT']['price'])

#Getting market depth if you are in BTC !
depth = client.get_order_book(symbol='BTCUSDT')

#depth
# getting all bids with last updated time

depth_df = pd.DataFrame(depth['asks'])
depth_df.columns = ['Price', 'ASK_Volume']
#depth_df.head()

depth_df1 = pd.DataFrame(depth['bids'])
depth_df1.columns = ['Price', 'BIDS_Volume']
#depth_df1.head()

#client.get_historical_klines??

historical = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, '1 Jan 2011')
#historical = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_4HOUR, '1 Jan 2011')
#historical

#hist_df = pd.DataFrame(historical)
dataset = pd.DataFrame(historical).astype(np.float32)

dataset.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
                    'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

#dataset.isnull().sum()

dataset.drop(['Open Time', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'TB Base Volume', 'TB Quote Volume','Ignore'],1,inplace=True)
#dataset.drop(['Volume', 'Quote Asset Volume', 'Number of Trades', 'TB Base Volume', 'TB Quote Volume','Ignore'],1,inplace=True)

#df.drop(['Volume', 'Quote Asset Volume', 'Number of Trades', 'TB Base Volume', 'TB Quote Volume','Ignore'],1,inplace=True)
#dataset.info()
#df.head()
#dataset.tail()

dataset['OHLC_avg'] = dataset[['Open','High', 'Low', 'Close']].mean(axis = 1)
#dataset.tail(5)

# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset.mean(axis = 1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
close_val = dataset[['Close']]

# PREPARATION OF TIME SERIES DATASE
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

# TRAIN-TEST SPLIT
#train_OHLC = int(len(OHLC_avg) * 0.75)
train_OHLC = int(len(OHLC_avg) * 0.90)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)


# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1



### LOAD MODEL


#https://github.com/galichandrey/h5model/raw/main/myh5model.h5
#model = tf.keras.models.load_model('my_h5_model.h5')
!wget https://github.com/galichandrey/h5model/raw/main/myh5model.h5
model = tf.keras.models.load_model('myh5model.h5')

# Check its architecture
#model.summary()

# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

"""Для загрузки модели!!"""

#!pip install pyyaml h5py
#checkpoint_path = "cp-{epoch:04d}.ckpt"
#### Save the weights using the `checkpoint_path` format
#model.save_weights(checkpoint_path.format(epoch=5))

#model.save_weights("ckpt")
#load_status = model.load_weights("ckpt")

####todo: Save Checkpoint
#!mkdir -p saved_model
#model.save('saved_model/my_model')
###model.save("my_h5_model.h5")

##todo: Load Checkpoint
#!pip install pyyaml h5py
#import tensorflow as tf
#model = tf.keras.models.load_model('my_h5_model.h5')

## Check its architecture
###model.summary()

# DE-NORMALIZING MAIN DATASET 
OHLC_avg = scaler.inverse_transform(OHLC_avg)

# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print ('Last Day Value:', last_val.item())

#print "Next Day Value:", np.asscalar(last_val*next_val)
#print ("Close Price right now: ", float(dataset['Close'].tail(1).values))
print ("Close Price right now: ", float(dataset['Close'].tail(1)))
nextday = last_val.item()*next_val.item()
print ('Next Day Value:', nextday)
