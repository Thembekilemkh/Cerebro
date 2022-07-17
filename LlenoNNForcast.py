from cmath import log
from locale import currency
from sys import dont_write_bytecode
from time import sleep
from tracemalloc import start
from webbrowser import get
from matplotlib.style import available
import requests
from xml.dom.expatbuilder import theDOMImplementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygments import highlight
plt.style.use("fivethirtyeight")
from sklearn.preprocessing import MinMaxScaler
#import keras
from keras import models
from keras import layers
from keras.models import load_model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
from matplotlib.dates import date2num
from datetime import timedelta, date
from datetime import datetime
import datetime as dt
from tqdm import trange
import duka.app.app as import_ticks
from duka.core.utils import TimeFrame
import timeit
import threading
from queue import Queue
import pickle
import json
import pytz
import os

import csv

# Custom Libraries
from tradesystems import MultiTradeSystem
from tradesystems import TradeSystem

class HourATP():
    def __init__(self, **kwargs):
        timeframe = kwargs['timeframe']
        location = kwargs['location']
        csv_file = kwargs['csv_file']
        apply_to = kwargs['apply_to']

        tikData = pd.read_csv(location + csv_file, parse_dates=[0])
        tikData.columns = [["time", "open", "high", "low", "close", "volume"]]
        self.c_Pair = self.csv_File[0:6]

        timeData = self.fix_dates(tikData=tikData)

        array_set = self.get_time_arrays(timeframe=timeframe, data=tikData, timedata=timeData,
                             apply_to=apply_to)
    
    def Average_True_Range(self, **kwargs):
        def True_Range(date, closep, highp, lowp, openp, yester_close):

            x = highp-lowp
            y = abs(highp - yester_close)
            z = abs(lowp - yester_close)

            if y <= x > z:
                TR = x

            elif x <= y >= z:
                TR = y

            elif x <= z >= y:
                TR=z

            return date, TR
            
        price_data = kwargs['data']
        period = int(kwargs['period'])
        price_data.columns = ["date", "open", "close", "high", "low"]
        date = price_data['date'].to_numpy()
        openp = price_data['open'].to_numpy()
        highp = price_data['high'].to_numpy()
        lowp = price_data['low'].to_numpy()
        closep = price_data['close'].to_numpy()
        
        #Calling the ATR
            
        x = 1
        TRDates = []
        TrueRanges = []
        ATR = []

        while x < len(date):
            TRDate, TrueRange = True_Range(date[x], closep[x], highp[x], lowp[x], openp[x], closep[x-1])

            TRDates.append(TRDate)
            TrueRanges.append(TrueRange)

            if x > period+1:
                trData = TrueRanges[x-(period+1):x]
                
                atr = self.ema(self, data=trData, period=period)
                ATR.append(atr[0])
            x = x+1
        return ATR, "off"

    def get_time_arrays(self, **kwargs):
        timeframe = kwargs['timeframe']
        data = kwargs['data']
        timedata = kwargs['timedata']
        apply_to = kwargs['apply_to']
        delta = self.get_timedelta(timeframe=timeframe)
        grouped_dict = {}

        for x in range(len(timedata)):
            if timeframe == "1H" or timeframe == "4H":
                hr = timedata.iloc[x].hour 
                '''
                df = pd.DataFrame(list(zip(lst, lst2)),
               columns =['Name', 'val'])'''
                key = f'hour{hr}'
                
                if grouped_dict.has_key(key):
                    hour_data_c = data['close'].iloc[x]
                    hour_data_l = data['low'].iloc[x]
                    hour_data_h = data['high'].iloc[x]
                    hour_data_o = data['open'].iloc[x]

                    grouped_dict[key]['close'].append(hour_data_c)
                    grouped_dict[key]['open'].append(hour_data_o)
                    grouped_dict[key]['high'].append(hour_data_h)
                    grouped_dict[key]['low'].append(hour_data_l)
                else:
                    grouped_dict[key] = {}
                    grouped_dict[key]['close'] = []
                    grouped_dict[key]['open'] = []
                    grouped_dict[key]['high'] = []
                    grouped_dict[key]['low'] = []

                    hour_data_c = data['close'].iloc[x]
                    hour_data_l = data['low'].iloc[x]
                    hour_data_h = data['high'].iloc[x]
                    hour_data_o = data['open'].iloc[x]

                    grouped_dict[key]['close'].append(hour_data_c)
                    grouped_dict[key]['open'].append(hour_data_o)
                    grouped_dict[key]['high'].append(hour_data_h)
                    grouped_dict[key]['low'].append(hour_data_l)

            elif timeframe == "1D":
                day = timedata.iloc[x].day 
                key = f'day{day}'
                
                if grouped_dict.has_key(key):
                    day_data_c = data['close'].iloc[x]
                    day_data_l = data['low'].iloc[x]
                    day_data_h = data['high'].iloc[x]
                    day_data_o = data['open'].iloc[x]

                    grouped_dict[key]['close'].append(day_data_c)
                    grouped_dict[key]['open'].append(day_data_o)
                    grouped_dict[key]['high'].append(day_data_h)
                    grouped_dict[key]['low'].append(day_data_l)
                else:

                    grouped_dict[key] = {}
                    grouped_dict[key]['close'] = []
                    grouped_dict[key]['open'] = []
                    grouped_dict[key]['high'] = []
                    grouped_dict[key]['low'] = []

                    day_data_c = data['close'].iloc[x]
                    day_data_l = data['low'].iloc[x]
                    day_data_h = data['high'].iloc[x]
                    day_data_o = data['open'].iloc[x]

                    grouped_dict[key]['close'].append(day_data_c)
                    grouped_dict[key]['open'].append(day_data_o)
                    grouped_dict[key]['high'].append(day_data_h)
                    grouped_dict[key]['low'].append(day_data_l)
        
        return grouped_dict
        
    def get_timedelta(self, **kwargs):
        timeframe = kwargs["timeframe"]

        if timeframe == "1H":
            delta = timedelta(hours=1)
        elif timeframe == "4H":
            delta = timedelta(hours=4)
        elif timeframe == "1D":
            delta = timedelta(days=1)

        return delta
            

    def fix_dates(self, **kwargs):
        tikData = kwargs['tikData']

        # Fixing the month and day swap issue
        print('\nCleaning dates...\n')
        new_dates = []

        if list(tikData[['time']].iloc[0])[0].day >= 13:
            switched = False
        else:
            switched = True

        prev_month = 0
        prev_day = 0
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        current_month = ''

        #for i in range(len(tikData[['time']])):
        for i in trange(len(tikData[['time']]), desc="Fixing dates"):
            # Gather old date variables
            tmp_date = list(tikData[['time']].iloc[i])[0]
            tmp_day = tmp_date.day
            tmp_year = tmp_date.year
            tmp_month = tmp_date.month
            tmp_hour = tmp_date.hour
            tmp_minute = tmp_date.minute
            tmp_second = tmp_date.second
            tmp_microsecond = tmp_date.microsecond

            '''
            if tmp_year == 2003:
                if switched == False:
                    print(f'{tmp_year}-{tmp_month}-{tmp_day}    {current_month}    {prev_day}     {prev_month}         {switched}')
                else:
                    print(f'{tmp_year}-{tmp_day}-{tmp_month}    {current_month}    {prev_day}     {prev_month}         {switched}')
            '''
            

            if switched:
                if tmp_day >= 13:#and prev_month == 12
                    switched = False
                    new_date = datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_month-1]
                    prev_month = tmp_month
                    prev_day = tmp_day
                else:
                    new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_day-1]
                    prev_month = tmp_day
                    prev_day = tmp_month
            else:
                if tmp_month == 1 and prev_day == 31 and (current_month == 'Jan' or current_month == 'Mar'or current_month == 'May' or current_month == 'Jul' or current_month == 'Aug' or current_month == 'Oct' or current_month == 'Dec'):
                    if current_month != 'Jan':
                        switched = True
                        new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                        current_month = months[tmp_day-1]
                        prev_month = tmp_day
                        prev_day = tmp_month
                    else:
                        new_date = datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                        current_month = months[tmp_month-1]
                        prev_month = tmp_month
                        prev_day = tmp_day

                elif tmp_month == 1 and current_month == 'Feb' and (prev_day == 28 or prev_day == 29):
                    switched = True
                    new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_day-1]
                    prev_month = tmp_day
                    prev_day = tmp_month

                elif tmp_month == 1 and prev_day == 30 and (current_month == 'Apr' or current_month == 'Jun' or current_month == 'Sep' or current_month == 'Nov'):
                    #print(f"Nov switch: {tmp_month == 1 and prev_day == 30 and (current_month == 'Apr' or current_month == 'Jun' or current_month == 'Sep' or current_month == 'Nov')}")
                    switched = True
                    new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_day-1]
                    prev_month = tmp_day
                    prev_day = tmp_month

                else:
                    new_date = datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_month-1]
                    prev_month = tmp_month
                    prev_day = tmp_day

            # export new date
            new_dates.append(new_date)

        print('\nDates cleaned\n')

        return pd.DataFrame(new_dates, columns=["time"])


class LlenoNNForcast():
    def __init__(self, **kwargs):
        # Getting the data we gonna make predictions
        function = kwargs['function']
        if function == 'build':
            self.location = kwargs["location"]
            self.location1 = kwargs["location1"]
            self.location2 = kwargs["location2"]

            self.csv_File = kwargs['csv_file']
            self.csv_File1 = kwargs['csv_file1']
            self.csv_File2 = kwargs['csv_file2']

            self.apply_to = kwargs['apply_to']
            self.look_back = kwargs["look_back"]
            self.cp_Path = kwargs['cp_Path']
            timeQ = kwargs["timeQ"]
            last_closeQ = kwargs["last_closeQ"]
            #self.timeframe = kwargs['time_frame']

            # Import lowest timeframe data
            print("\nImporting data files\n")
            print(self.location + self.csv_File)
            #d_parser = lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S.%f %z') #%z
            tikData = pd.read_csv(self.location + self.csv_File, parse_dates=[0])
            tikData.columns = [["time", "open", "high", "low", "close", "volume"]]
            self.c_Pair = self.csv_File[0:6]

            # take the date and munipulate it yourself, the month and the day are switched
            timeData = self.fix_dates(tikData=tikData)
            #print(time_df[['time']].head(675))
            self.timeData = timeData.to_numpy()

            data = tikData[[apply_to]]
            self.data = data.to_numpy()

            # Import middle timeframe data
            print(self.location1 + self.csv_File1)
            tikData = pd.read_csv(self.location1 + self.csv_File1, parse_dates=[0])
            tikData.columns = [["time", "open", "high", "low", "close", "volume"]]
            self.c_Pair1 = self.csv_File1[0:6]

            #timeData1 = tikData[['time']]
            timeData1 = self.fix_dates(tikData=tikData)
            self.timeData1 = timeData1.to_numpy()

            data1 = tikData[[apply_to]]
            self.data1 = data1.to_numpy()

            # Import highest timeframe data
            print(self.location2 + self.csv_File2)
            tikData = pd.read_csv(self.location2 + self.csv_File2, parse_dates=[0])
            tikData.columns = [["time", "open", "high", "low", "close", "volume"]]
            self.c_Pair2 = self.csv_File2[0:6]

            #timeData2 = tikData[['time']]
            timeData2 = self.fix_dates(tikData=tikData)
            self.timeData2 = timeData2.to_numpy()

            data2 = tikData[[apply_to]]
            self.data2 = data2.to_numpy()

            timeQ.put(timeData.iloc[-1])
            timeQ.put(timeData1.iloc[-1])
            timeQ.put(timeData2.iloc[-1])

            last_closeQ.put(data.iloc[-1])
            last_closeQ.put(data1.iloc[-1])
            last_closeQ.put(data2.iloc[-1])

        elif function == 'predict':
            self.location = kwargs["location"]
            self.csv_File = kwargs['csv_file']
            self.apply_to = kwargs['apply_to']
            self.look_back = kwargs["look_back"]
            self.h5model = kwargs['h5model']
            self.c_Pair = self.csv_File[0:6]

    def fix_dates(self, **kwargs):
        tikData = kwargs['tikData']

        # Fixing the month and day swap issue
        print('\nCleaning dates...\n')
        new_dates = []

        if list(tikData[['time']].iloc[0])[0].day >= 13:
            switched = False
        else:
            switched = True

        prev_month = 0
        prev_day = 0
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        current_month = ''

        #for i in range(len(tikData[['time']])):
        for i in trange(len(tikData[['time']]), desc="Fixing dates"):
            # Gather old date variables
            tmp_date = list(tikData[['time']].iloc[i])[0]
            tmp_day = tmp_date.day
            tmp_year = tmp_date.year
            tmp_month = tmp_date.month
            tmp_hour = tmp_date.hour
            tmp_minute = tmp_date.minute
            tmp_second = tmp_date.second
            tmp_microsecond = tmp_date.microsecond

            '''
            if tmp_year == 2003:
                if switched == False:
                    print(f'{tmp_year}-{tmp_month}-{tmp_day}    {current_month}    {prev_day}     {prev_month}         {switched}')
                else:
                    print(f'{tmp_year}-{tmp_day}-{tmp_month}    {current_month}    {prev_day}     {prev_month}         {switched}')
            '''
            

            if switched:
                if tmp_day >= 13:#and prev_month == 12
                    switched = False
                    new_date = datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_month-1]
                    prev_month = tmp_month
                    prev_day = tmp_day
                else:
                    new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_day-1]
                    prev_month = tmp_day
                    prev_day = tmp_month
            else:
                if tmp_month == 1 and prev_day == 31 and (current_month == 'Jan' or current_month == 'Mar'or current_month == 'May' or current_month == 'Jul' or current_month == 'Aug' or current_month == 'Oct' or current_month == 'Dec'):
                    if current_month != 'Jan':
                        switched = True
                        new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                        current_month = months[tmp_day-1]
                        prev_month = tmp_day
                        prev_day = tmp_month
                    else:
                        new_date = datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                        current_month = months[tmp_month-1]
                        prev_month = tmp_month
                        prev_day = tmp_day

                elif tmp_month == 1 and current_month == 'Feb' and (prev_day == 28 or prev_day == 29):
                    switched = True
                    new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_day-1]
                    prev_month = tmp_day
                    prev_day = tmp_month

                elif tmp_month == 1 and prev_day == 30 and (current_month == 'Apr' or current_month == 'Jun' or current_month == 'Sep' or current_month == 'Nov'):
                    #print(f"Nov switch: {tmp_month == 1 and prev_day == 30 and (current_month == 'Apr' or current_month == 'Jun' or current_month == 'Sep' or current_month == 'Nov')}")
                    switched = True
                    new_date = datetime(tmp_year,tmp_day,tmp_month,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_day-1]
                    prev_month = tmp_day
                    prev_day = tmp_month

                else:
                    new_date = datetime(tmp_year,tmp_month,tmp_day,tmp_hour,tmp_minute,tmp_second,tmp_microsecond)
                    current_month = months[tmp_month-1]
                    prev_month = tmp_month
                    prev_day = tmp_day

            # export new date
            new_dates.append(new_date)

        print('\nDates cleaned\n')

        return pd.DataFrame(new_dates, columns=["time"])

    def preprocess(self, **kwargs):
        # Getting training data size
        train_size = int(len(self.data) * 0.8)
        train_size1 = int(len(self.data1) * 0.8)
        train_size2 = int(len(self.data2) * 0.8)


        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)
        scaled_data1 = scaler.fit_transform(self.data1)
        scaled_data2 = scaler.fit_transform(self.data2)

        # Creating the train dataset
        train_data = scaled_data[0:train_size, :]
        train_data1 = scaled_data1[0:train_size1, :]
        train_data2 = scaled_data2[0:train_size2, :]

        # Split into the training data and the actual data
        x_train = []
        y_train = []
        look_back = 60
        for i in trange(look_back, len(train_data), desc="Splitting training data"):
            x_train.append(train_data[i - 60:i, :])
            y_train.append(train_data[i, :])


        # Split into the training data and the actual data
        x_train1 = []
        y_train1 = []
        look_back = 60
        for i in trange(look_back, len(train_data1), desc="Splitting middle timeframe training data"):
            x_train1.append(train_data1[i - 60:i, :])
            y_train1.append(train_data1[i, :])

        # Split into the training data and the actual data
        x_train2 = []
        y_train2 = []
        look_back = 60
        for i in trange(look_back, len(train_data2), desc="Splitting frame highest timetraining data"):
            x_train2.append(train_data2[i - 60:i, :])
            y_train2.append(train_data2[i, :])

        # Convert x_train and y_train to a numpy array
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
        x_train2, y_train2 = np.array(x_train2), np.array(y_train2)

        # Preparing variables for our LSTM
        features = 1
        rows = x_train.shape[0]
        timesteps = x_train.shape[1]
        batch = 1  # 1 2 1459 2918

        # Middle timeframe
        rows1 = x_train1.shape[0]
        timesteps1 = x_train1.shape[1]

        # Highest timeframe
        rows2 = x_train2.shape[0]
        timesteps2 = x_train2.shape[1]

        # reshape data to a 3D array
        x_train = np.reshape(x_train, (rows, timesteps, features))
        x_train1 = np.reshape(x_train1, (rows1, timesteps1, features))
        x_train2 = np.reshape(x_train2, (rows2, timesteps2, features))

        pre_processed_data = {"lowest_timeframe":[x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data],
                              "middle_timeframe":[x_train1, y_train1, batch, train_size1, timesteps1, features, scaler, scaled_data1],
                              "highest_timeframe":[x_train2, y_train2, batch, train_size2, timesteps2, features, scaler, scaled_data2]}
        return pre_processed_data

    def build_model(self, **kwargs):
        # Get required data
        timesteps = kwargs["timesteps"]
        features = kwargs["features"]

        # Build LSTM model
        model = Sequential()

        model.add(LSTM(
            50, activation="relu",
            return_sequences=True, input_shape=(timesteps, features)))  # stateful=True,
        model.add(layers.Dropout(0.2))

        model.add(LSTM(100, activation="relu",
                       return_sequences=True))  # stateful=True
        model.add(layers.Dropout(0.4))

        model.add(layers.LSTM(50, activation="relu",
                              return_sequences=False))  # stateful=True
        model.add(layers.Dropout(0.2))

        model.add(Dense(25, activation="relu"))
        model.add(Dense(1, activation="relu"))

        return model

    def compileNsave(self, **kwargs):
        # Get relavent data
        model = kwargs['model']
        x_train = kwargs['x_train']
        y_train = kwargs['y_train']
        batch = kwargs['batch']
        timeframe=kwargs['timeframe']
        system_name=kwargs['system_name']

        # Compile model
        model.compile(optimizer="adam", loss="mean_squared_error",
                      metrics=["accuracy"])

        # Saves the model every now and again incase something happens
        checkpoint = ModelCheckpoint(self.cp_Path, save_best_only=True)

        # Getting our model ready to stop early if it stops improving
        ES = EarlyStopping()

        # Train the model
        # for i in trange(1, desc="Training the fuck out of this model"):
        model.fit(x_train, y_train,
                  batch_size=batch, epochs=1,
                  callbacks=[checkpoint, ES])  # shuffle=False
        # model.reset_states()

        # Saving the model for later use
        #print("Model created")
        model.save( + f"{system_name} {self.c_Pair}_lleno_{self.apply_to}_{timeframe}.h5")

        return self.c_Pair

    def test_model(self, **kwargs):
        # Get required data
        train_size = kwargs['train_size']
        model = kwargs['model']
        batch = kwargs['batch']
        scaled_data = kwargs["scaled_data"]
        look_back = kwargs["look_back"]
        scaler = kwargs["scaler"]
        frame = kwargs["frame"]
        q = kwargs['Q']

        #print(f"\nTrain size: {train_size}\n")

        if frame == "lower":
            data = self.data
        elif frame == "middle":
            data = self.data1
        elif frame == "higher":
            data = self.data2

        # Create the testing data
        test_data = scaled_data[train_size - look_back: len(data), :]

        # Split into the test data and the actual data
        x_test = []
        y_test = data[train_size:, :]

        for z in trange(look_back, len(test_data), desc="Splitting test data"):
            x_test.append(test_data[z - look_back:z, :])

        # Convert x_test and y_test to a numpy array
        x_test = np.array(x_test)

        # Preparing test data
        test_features = 1
        test_timesteps = x_test.shape[1]
        test_rows = x_test.shape[0]

        # Reshape the data into a 3D array
        x_test = np.reshape(x_test, (test_rows, test_timesteps, test_features))

        # Get the predicted price values (for the x_test data set)
        predictions = model.predict(x_test, batch_size=batch)

        # Unscale (Inverse transform the data)
        predictions = scaler.inverse_transform(predictions)
        predict_df = pd.DataFrame(data=predictions, columns=["Predictions"])
        #print(predict_df)
        #print(type(predict_df))

        # Model evaluation getting the root mean squared error
        mse = np.sqrt(np.mean(predictions - y_test) ** 2)
        #print(mse)

        q.put(mse)
        q.put(predict_df)

    def plot_training(self, **kwargs):
        # Get required data
        train_size = kwargs['train_size']
        apply_to = kwargs['apply_to']
        predict_df = kwargs['predict_df']
        title = kwargs["title"]

        # Getting data ready to be plotted
        train = self.data[:train_size]
        train_time_data = self.timeData[:train_size]
         
        train_df = pd.DataFrame(data=train, columns=[apply_to])

        print(train_df.iloc[[0]])
        time_df = pd.DataFrame(data=train_time_data, columns=['Time'])
        #print(f"time:{time_df.iloc[[0]]}")
        #time_df.strftime(%Y-%m-%d %H:%M:%S.%f)
        #time_df = pd.to_datetime(time_df, infer_datetime_format=True)#format='%Y-%m-%d %H:%M:%S.%f'

        valid = self.data[train_size:]
        train_time_data = self.timeData[train_size:]

        valid_df = pd.DataFrame(data=valid, columns=[apply_to])
        time_df2 = pd.DataFrame(data=train_time_data, columns=["Time"])
        #time_df2 = pd.to_datetime(time_df2, format='%Y-%m-%d %H:%M:%S.%f')

        valid_df["Predictions"] = predict_df[["Predictions"]]

        print(f"train data: {time_df.head()}")
        print(f"valid data: {time_df2.head()}")
        # plot the data
        plt.figure(figsize=(16, 8))
        plt.title(title)
        plt.xlabel("Date", fontsize=18)
        plt.ylabel(f"{apply_to.capitalize()} price", fontsize=18)
        plt.plot(time_df[["Time"]],train_df[[apply_to]])
        plt.plot(time_df2[["Time"]], valid_df[[apply_to, "Predictions"]])
        plt.legend(["Training data", "Value", "Prediction data"], loc="lower right")
        plt.show()

        # Show the valid price and predicted price
        #print(valid_df)

    def predict_future(self, **kwargs):
        location = kwargs['location']
        csv_File = kwargs['csv_file']
        apply_to = kwargs['apply_to']
        look_back = kwargs['look_back']

        model = load_model(self.h5model)

        
        tikData = pd.read_csv(location + csv_File, parse_dates=[0])
        tikData.columns = [["time", "open", "high", "low", "close", "volume"]]
        c_Pair = csv_File[0:6]
        data = tikData[[apply_to]]

        # Date preprocessing
        date = tikData[["time"]]  # Date type:  <class 'pandas._libs.tslibs.timestamps.Timestamp'>
        date = date.to_numpy()
        # date = date2num(date)
        print("Indexed")

        # Get the last data the length of the look back period
        last_look_back = data[-look_back:]
        date = date[-look_back:]
        data = last_look_back.to_numpy()

        # Scale data between 0 and 1
        print(f"Shape before being shaped: {data.shape}")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create an empty list
        X_data = []
        predicted_data = []
        predicted_dates = []
        all_data = []
        all_dates = []

        # Append last look back data
        X_data.append(scaled_data)

        # Prepare a list that will contain all the data
        for y in range(len(scaled_data)):
            all_data.append(scaled_data[y][0])
            all_dates.append(date[y][0].to_pydatetime())

        # Convert to numpy
        X_data = np.array(X_data)
        dates = np.array(all_dates)
        dates = date2num(dates)
        original_data = X_data
        current_date = all_dates[-1]
        predicted_dates.append(current_date)

        # Loop through and keep getting new predicted data
        x = 0
        while x < 120:
            # Reshape to 3D Array
            X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

            # Get predicted scaled price
            predicted_price = model.predict(X_data)

            # Increment dates
            current_date = current_date + timedelta(hours=1)

            # Append predicted data where needed
            X_data = np.append(X_data, predicted_price[0][0])
            predicted_data.append(predicted_price[0][0])
            all_data.append(predicted_price[0][0])
            predicted_dates.append(current_date)

            # Get the last look back and prepare data for next round
            X_data = X_data[-look_back:]

            X_data = np.reshape(X_data, (1, X_data.shape[0], 1))

            # Add one hour to the date array

            x = x + 1

        print(f"\nData shape: {X_data.shape}")
        print(f"Our predicted date: {predicted_dates[0]}")
        X_data = np.reshape(X_data, (X_data.shape[1]))

        # Turn to numpy and shape data to be reshaped
        prediction = all_data[-121:]
        prediction = np.array(prediction)
        prediction = np.reshape(prediction, (prediction.shape[0], 1))
        num_all = np.array(all_data)
        num_all = np.reshape(num_all, (num_all.shape[0], 1))
        original_data = np.reshape(original_data, (original_data.shape[1], 1))
        predicted_dates = np.array(predicted_dates)
        predicted_dates = date2num(predicted_dates)

        # Inverse transform
        real_data = scaler.inverse_transform(num_all)
        original_data = scaler.inverse_transform(original_data)
        prediction = scaler.inverse_transform(prediction)

        return original_data, dates, prediction, predicted_dates, real_data

    def plot_future(self, **kwargs):
        # Get required data
        original_data = kwargs["original_data"]
        dates = kwargs['dates']
        prediction = kwargs['prediction']
        predicted_dates = kwargs['predicted_dates']
        real_data = kwargs['real_data']

        # Getting data ready to be plotted
        initial_df = pd.DataFrame(data=original_data, columns=["original"])
        ini_date_df = pd.DataFrame(data=dates, columns=["date"])
        prediction_df = pd.DataFrame(data=prediction, columns=["prediction_data"])
        predict_date_df = pd.DataFrame(data=predicted_dates, columns=["new_date"])
        new_df = pd.DataFrame(data=real_data, columns=["new_data"])

        print(f"Predictions: {prediction_df}")
        # Visualize the model

        plt.figure(figsize=(16, 8))
        plt.title("Market forecast")
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Close price", fontsize=18)
        plt.plot(ini_date_df[["date"]], initial_df[["original"]])
        plt.plot(predict_date_df[["new_date"]], prediction_df[["prediction_data"]])
        plt.legend(["Original data", "Predictions for next 5 days"], loc="lower right")
        plt.show()

    def get_latest_timestamps(self, **kwargs):
        location = kwargs["location"]
        file = kwargs["file"]
        
        
        print("\nImporting data files\n")
        tikData = pd.read_csv(location + file, parse_dates=[0])
        tikData.columns = [["time", "open", "high", "low", "close", "volume"]]

        timedata = tikData[["time"]]
        #print(f"Last time stamp: {timedata.iloc[-1] + timedelta(hours=1)}")



def train_models(pre_processed_data, forcast_obj, que, fileQ, system_name):
    # x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data, model
    file = fileQ.get()
    file1 = fileQ.get()
    file2 = fileQ.get()

    c_pair = forcast_obj.compileNsave(model=pre_processed_data['lowest_timeframe'][8], x_train=pre_processed_data['lowest_timeframe'][0],
                                           y_train=pre_processed_data['lowest_timeframe'][1], batch=pre_processed_data['lowest_timeframe'][2], 
                                           timeframe=get_timeframe(file), system_name=system_name)
    pre_processed_data['lowest_timeframe'].append(c_pair)

    c_pair = forcast_obj.compileNsave(model=pre_processed_data['middle_timeframe'][8], x_train=pre_processed_data['middle_timeframe'][0],
                                           y_train=pre_processed_data['middle_timeframe'][1], batch=pre_processed_data['middle_timeframe'][2],
                                           timeframe=get_timeframe(file1), system_name=system_name)
    pre_processed_data['middle_timeframe'].append(c_pair)

    c_pair = forcast_obj.compileNsave(model=pre_processed_data['highest_timeframe'][8], x_train=pre_processed_data['highest_timeframe'][0],
                                           y_train=pre_processed_data['highest_timeframe'][1], batch=pre_processed_data['highest_timeframe'][2],
                                           timeframe=get_timeframe(file2), system_name=system_name)
    pre_processed_data['highest_timeframe'].append(c_pair)


    # x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data, model, c_pair

    que.put(pre_processed_data)


def test_models(pre_processed_data, test_func, que, look_back):
    lowerQ = Queue()
    midQ = Queue()
    higherQ = Queue()
    # x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data, model, c_pair
    """threading.Thread(target=test_func, kwargs={"train_size":pre_processed_data['lowest_timeframe'][3], 
                                               "model":pre_processed_data['lowest_timeframe'][8], 
                                               "scaler":pre_processed_data['lowest_timeframe'][6],
                                               "batch":pre_processed_data['lowest_timeframe'][2], 
                                               "scaled_data":pre_processed_data['lowest_timeframe'][7], 
                                               "look_back":look_back, "frame":"lower", "Q":lowerQ}).start()"""
    
    test_func(train_size=pre_processed_data['lowest_timeframe'][3],
              model=pre_processed_data['lowest_timeframe'][8],
              scaler=pre_processed_data['lowest_timeframe'][6],
              batch=pre_processed_data['lowest_timeframe'][2],
              scaled_data=pre_processed_data['lowest_timeframe'][7],
              look_back=look_back, frame="lower", Q=lowerQ)
    pre_processed_data['lowest_timeframe'].append(lowerQ.get())
    pre_processed_data['lowest_timeframe'].append(lowerQ.get())

    """threading.Thread(target=test_func, kwargs={"train_size":pre_processed_data['middle_timeframe'][3], 
                                               "model":pre_processed_data['middle_timeframe'][8], 
                                               "scaler":pre_processed_data['middle_timeframe'][6],
                                               "batch":pre_processed_data['middle_timeframe'][2], 
                                               "scaled_data":pre_processed_data['middle_timeframe'][7], 
                                               "look_back":look_back, "frame":"lower", "Q":midQ}).start()"""

    test_func(train_size=pre_processed_data['middle_timeframe'][3],
              model=pre_processed_data['middle_timeframe'][8],
              scaler=pre_processed_data['middle_timeframe'][6],
              batch=pre_processed_data['middle_timeframe'][2],
              scaled_data=pre_processed_data['middle_timeframe'][7],
              look_back=look_back, frame="middle", Q=midQ)    
    pre_processed_data['middle_timeframe'].append(midQ.get())
    pre_processed_data['middle_timeframe'].append(midQ.get())


    """threading.Thread(target=test_func, kwargs={"train_size":pre_processed_data['highest_timeframe'][3], 
                                               "model":pre_processed_data['highest_timeframe'][8], 
                                               "scaler":pre_processed_data['highest_timeframe'][6],
                                               "batch":pre_processed_data['highest_timeframe'][2], 
                                               "scaled_data":pre_processed_data['highest_timeframe'][7], 
                                               "look_back":look_back, "frame":"lower", "Q":higherQ}).start()"""
    
    test_func(train_size=pre_processed_data['highest_timeframe'][3],
              model=pre_processed_data['highest_timeframe'][8],
              scaler=pre_processed_data['highest_timeframe'][6],
              batch=pre_processed_data['highest_timeframe'][2],
              scaled_data=pre_processed_data['highest_timeframe'][7],
              look_back=look_back, frame="higher", Q=higherQ)
    pre_processed_data['highest_timeframe'].append(higherQ.get())
    pre_processed_data['highest_timeframe'].append(higherQ.get())

    # x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data, model, c_pair, mse, predict_df
    que.put(pre_processed_data)

def get_timeframe(file):
    got_currency = False
    got_format = False
    got_timeframe = False
    got_break = False
    timeframe = ""
    for l in file:
        if got_currency == False:
            if l == "_":
                got_currency = True

        elif got_format == False:
            if l == "_":
                got_format = True

        elif got_timeframe == False:
            if l == "_":
                if got_break == True:
                    got_timeframe = True
                    break
                else:
                    got_break = True
            else:
                timeframe = timeframe+l
                

    return timeframe

def convert_to_dict(pre_processed_data):
    # x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data, model, c_pair, mse, predict_df, timeframe
    low_dict = {}
    mid_dict = {}
    high_dict = {}
    for key, val in pre_processed_data.items():
        if key == "lowest_timeframe":
            for v in range(len(val)):
                if v == 0:
                    low_dict["x_train"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 1:
                    low_dict["y_train"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 2:
                    low_dict["batch"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 3:
                    low_dict["train_size"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 4:
                    low_dict["timesteps"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 5:
                    low_dict["features"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 6:
                    low_dict["scaler"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 7:
                    low_dict["scaled_data"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 8:
                    low_dict["model"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 9:
                    low_dict["c_pair"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 10:
                    low_dict["mse"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 11:
                    low_dict["predict_df"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 12:
                    low_dict["timeframe"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 13:
                    low_dict["last_trained_time"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 14:
                    low_dict["last_close"] = pre_processed_data['lowest_timeframe'][v]
                elif v == 15:
                    low_dict["next_close_date"] = pre_processed_data['lowest_timeframe'][v]

            pre_processed_data['lowest_timeframe'] = low_dict

        if key == "middle_timeframe":
            for v in range(len(val)):
                if v == 0:
                    mid_dict["x_train"] = pre_processed_data['middle_timeframe'][v]
                elif v == 1:
                    mid_dict["y_train"] = pre_processed_data['middle_timeframe'][v]
                elif v == 2:
                    mid_dict["batch"] = pre_processed_data['middle_timeframe'][v]
                elif v == 3:
                    mid_dict["train_size"] = pre_processed_data['middle_timeframe'][v]
                elif v == 4:
                    mid_dict["timesteps"] = pre_processed_data['middle_timeframe'][v]
                elif v == 5:
                    mid_dict["features"] = pre_processed_data['middle_timeframe'][v]
                elif v == 6:
                    mid_dict["scaler"] = pre_processed_data['middle_timeframe'][v]
                elif v == 7:
                    mid_dict["scaled_data"] = pre_processed_data['middle_timeframe'][v]
                elif v == 8:
                    mid_dict["model"] = pre_processed_data['middle_timeframe'][v]
                elif v == 9:
                    mid_dict["c_pair"] = pre_processed_data['middle_timeframe'][v]
                elif v == 10:
                    mid_dict["mse"] = pre_processed_data['middle_timeframe'][v]
                elif v == 11:
                    mid_dict["predict_df"] = pre_processed_data['middle_timeframe'][v]
                elif v == 12:
                    mid_dict["timeframe"] = pre_processed_data['middle_timeframe'][v]
                elif v == 13:
                    mid_dict["last_trained_time"] = pre_processed_data['middle_timeframe'][v]
                elif v == 14:
                    low_dict["last_close"] = pre_processed_data['middle_timeframe'][v]
                elif v == 15:
                    low_dict["next_close_date"] = pre_processed_data['middle_timeframe'][v]

            pre_processed_data['middle_timeframe'] = mid_dict

        elif key == "highest_timeframe":
            for v in range(len(val)):
                if v == 0:
                    high_dict["x_train"] = pre_processed_data['highest_timeframe'][v]
                elif v == 1:
                    high_dict["y_train"] = pre_processed_data['highest_timeframe'][v]
                elif v == 2:
                    high_dict["batch"] = pre_processed_data['highest_timeframe'][v]
                elif v == 3:
                    high_dict["train_size"] = pre_processed_data['highest_timeframe'][v]
                elif v == 4:
                    high_dict["timesteps"] = pre_processed_data['highest_timeframe'][v]
                elif v == 5:
                    high_dict["features"] = pre_processed_data['highest_timeframe'][v]
                elif v == 6:
                    high_dict["scaler"] = pre_processed_data['highest_timeframe'][v]
                elif v == 7:
                    high_dict["scaled_data"] = pre_processed_data['highest_timeframe'][v]
                elif v == 8:
                    high_dict["model"] = pre_processed_data['highest_timeframe'][v]
                elif v == 9:
                    high_dict["c_pair"] = pre_processed_data['highest_timeframe'][v]
                elif v == 10:
                    high_dict["mse"] = pre_processed_data['highest_timeframe'][v]
                elif v == 11:
                    high_dict["predict_df"] = pre_processed_data['highest_timeframe'][v]
                elif v == 12:
                    high_dict["timeframe"] = pre_processed_data['highest_timeframe'][v]
                elif v == 13:
                    high_dict["last_trained_time"] = pre_processed_data['highest_timeframe'][v]
                elif v == 14:
                    low_dict["last_close"] = pre_processed_data['highest_timeframe'][v]
                elif v == 15:
                    low_dict["next_close_date"] = pre_processed_data['highest_timeframe'][v]

            pre_processed_data['highest_timeframe'] = high_dict

    print(f"History keys: {pre_processed_data.keys()}")

    return pre_processed_data

def get_available_systems():

    named_sys = []
    for file in os.listdir():
        if " " in file:
            if file[-2:] == "h5":
                name_found = False
                l = 0
                while name_found == False:
                    if file[-l] == " ":
                        name_found = True
                        if file[:-l] not in named_sys:
                            named_sys.append(file[:-l])
                    l = l+1
    return named_sys

if __name__ == '__main__':
    keep_looping = True
    while keep_looping:
        got_func = False
####################################################################################################################
        try:
            buildorpredict = int(input("1. Build model\n2. Start trade system\n3. Choose a system to start\n4. Make a prediction\n"))
            if buildorpredict <= 3:
                got_func = True
                logger = logging.getLogger(__name__)
            else:
                print("Enter either '1', '2' or '3'!!")

        except Exception as e:
            print("Enter a number!!")
####################################################################################################################

        if got_func == True:
            if buildorpredict == 1:
                # Get locations
                location = (input('Enter the loction for the lowest timeframe you will be trading on data, enter 0 if data is in this dir...\n')).replace("\\", '\\\\')
                location1 = (input('Enter the loction for the middle timeframe you will be trading on data, enter 0 if data is in this dir...\n')).replace("\\", '\\\\')
                location2 = (input('Enter the loction for the highest timeframe you will be trading on data, enter 0 if data is in this dir...\n')).replace("\\", '\\\\')
                location = f'{location}\\\\'
                location1 = f'{location1}\\\\'
                location2 = f'{location2}\\\\'

                # Get CSV files
                csv_file = input('Enter the name of the file for the lowest timeframe...\n')
                csv_file1 = input('Enter the name of the file for the middle timeframe...\n')
                csv_file2 = input('Enter the name of the file for the highest timeframe...\n')
                
                # On what should the the system be applied to?
                apply_to = input('What should the be applied to?\n1. Open\n2. High\n3. Low\n4. Close\n')
                
                # Name your system
                system_name = input("What would you like to name the system?\n") 

                function = "build"
                cp_path = "Get current working directory"
                look_back = 60
                timeQ = Queue()
                l_closeQ = Queue()

                # Set timeframes to use.
                print('\nWe will be using the following timeframes to make predictions')
                print('Lowest timeframe: 1 Hour')
                print('Middle timeframe: 4 Hour')
                print('Highest timeframe: 1 Day\n')

                if apply_to=='1':
                    apply_to='open'
                elif apply_to=='2':
                    apply_to='high'
                elif apply_to=='3':
                    apply_to='low'
                elif apply_to=='4':
                    apply_to='close'

                # Send data to object
                lleno_forcast = LlenoNNForcast(location=location, csv_file=csv_file,
                                               apply_to=apply_to, function=function,
                                               cp_Path=cp_path, look_back=look_back,
                                               csv_file1=csv_file1, csv_file2=csv_file2,
                                               location1=location1, location2=location2, timeQ=timeQ,
                                               last_closeQ=l_closeQ)


                # Pre process all data
                pre_processed_data = lleno_forcast.preprocess()

                # Build model
                model = lleno_forcast.build_model(timesteps=pre_processed_data["lowest_timeframe"][4],
                                                  features=pre_processed_data["lowest_timeframe"][5])
                pre_processed_data["lowest_timeframe"].append(model)

                model = lleno_forcast.build_model(timesteps=pre_processed_data["middle_timeframe"][4],
                                                  features=pre_processed_data["middle_timeframe"][5])
                pre_processed_data["middle_timeframe"].append(model)

                model = lleno_forcast.build_model(timesteps=pre_processed_data["highest_timeframe"][4],
                                                  features=pre_processed_data["highest_timeframe"][5])
                pre_processed_data["highest_timeframe"].append(model)
                
                # Compile models and save
                print('\nTraining and saving the model\n') 
                train_que = Queue()
                fileQ = Queue()
                fileQ.put(csv_file)
                fileQ.put(csv_file1)
                fileQ.put(csv_file2)
                t = threading.Thread(target=train_models, args=(pre_processed_data, lleno_forcast, train_que, fileQ, system_name)).start()
                pre_processed_data = train_que.get()

                # Test models
                print("\nTesting the models\n")
                test_que = Queue()
                test_models(pre_processed_data, lleno_forcast.test_model, test_que, look_back)
                test_que.get()

                
                # x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data, model, c_pair, mse, predict_df

                # Plot findings for all

                lleno_forcast.plot_training(train_size=pre_processed_data['lowest_timeframe'][3], apply_to=apply_to,
                                            predict_df=pre_processed_data['lowest_timeframe'][11],
                                            title=pre_processed_data['lowest_timeframe'][9])

            
                lleno_forcast.plot_training(train_size=pre_processed_data['middle_timeframe'][3], apply_to=apply_to,
                                            predict_df=pre_processed_data['middle_timeframe'][11],
                                            title=pre_processed_data['middle_timeframe'][9])
                
                lleno_forcast.plot_training(train_size=pre_processed_data['highest_timeframe'][3], apply_to=apply_to,
                                            predict_df=pre_processed_data['highest_timeframe'][11],
                                            title=pre_processed_data['highest_timeframe'][9])


                # x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data, model, c_pair, mse, predict_df, timeframe, last_trained_time
                tf = get_timeframe(csv_file)
                pre_processed_data["lowest_timeframe"].append(tf)

                tf = get_timeframe(csv_file1)
                pre_processed_data["middle_timeframe"].append(tf)

                tf = get_timeframe(csv_file2)
                pre_processed_data["highest_timeframe"].append(tf)

                last_time_data = timeQ.get()
                last_time_data1 = timeQ.get()
                last_time_data2 = timeQ.get()
                last_close = l_closeQ.get()
                last_close1 = l_closeQ.get()
                last_close2 = l_closeQ.get()

                pre_processed_data["lowest_timeframe"].append(last_time_data)
                pre_processed_data["middle_timeframe"].append(last_time_data1)
                pre_processed_data["highest_timeframe"].append(last_time_data2)

                pre_processed_data["lowest_timeframe"].append(last_close)
                pre_processed_data["middle_timeframe"].append(last_close1)
                pre_processed_data["highest_timeframe"].append(last_close2)

                pre_processed_data["lowest_timeframe"].append(last_time_data+timedelta(hours=1))
                pre_processed_data["middle_timeframe"].append(last_time_data1+timedelta(days=1))
                pre_processed_data["highest_timeframe"].append(last_time_data2+timedelta(weeks=1))

                pre_processed_data = convert_to_dict(pre_processed_data)


                # Save all data for future reference in a pickle
                # lleno_forcast.get_latest_timestamps(location=location, file=csv_file)

                # Remove model from the dictionary as it can't be save
                pre_processed_data["lowest_timeframe"].pop("model", None)
                pre_processed_data["middle_timeframe"].pop("model", None)
                pre_processed_data["highest_timeframe"].pop("model", None)

                """with open("trained_history.json", "w") as f:
                    json.dump(pre_processed_data, f)
                    print("\nModel saved!!\n")"""

                pre_processed_data_ = {system_name:pre_processed_data}
                with open(f'{system_name} trained_history.pickle', "wb") as f:
                    pickle.dump(pre_processed_data_, f)
                    print("\nModel saved!!\n")

                print("\nDone\n")

            elif buildorpredict == 2:
                # Begin trade loop
                got_model = False
                try:
                    # Load pickle
                    with open('trained_history.pickle', "rb") as f:
                        pre_processed_data = pickle.load(f)
                        got_model=True

                except Exception as e:
                    print(f"\nFailed to get model due to: {e}\n")
                
                if got_model:
                    #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
                    TradeSystem(pre_processed_data=pre_processed_data)

                else:
                    print(f"fail to get model: {e}")
            elif buildorpredict ==3:
                
                # Find available bots and check which one the user wants to run
                available_systems = get_available_systems()
                
                i = 1
                u_query = 'Choose a system to run:\n'
                for system in available_systems:
                    u_query = u_query + f"{i}) {system}\n"
                    i=i+1

                
                # Multi chart
                try:
                    chosen_one = int(input(u_query))
                    
                    if chosen_one <= len(available_systems):
                        
                        # Begin trade loop
                        got_model = False
                        try:
                            # Load pickle
                            with open(f'{available_systems[chosen_one-1]} trained_history.pickle', "rb") as f:
                                pre_processed_data = pickle.load(f)
                                got_model=True

                        except Exception as e:
                            print(f"\nFailed to get model due to: {e}\n")
                        
                        if got_model:
                            #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
                            MultiTradeSystem(pre_processed_data=pre_processed_data, system_name=available_systems[chosen_one-1])

                        else:
                            print(f"fail to get model: {e}")
                except Exception as e:
                    print(f'Please choose the number that represent the system you would like to run!!')
                    
            elif buildorpredict == 4:
                location = (input('Enter the loction of the data, enter 0 if data is in this dir...\n')).replace("\\", "\\\\")
                location = f'{location}\\\\'
                csv_file = input('Enter the name of the file...\n')
                apply_to = input('What should the be applied to?\n1. Open\n2. High\n3. Low\n4. Close\n')
                h5model = input("Choose a saved model\n")
                function = "predict"
                look_back = 60
                print("The location: ", location)

                if apply_to=='1':
                    apply_to='open'
                elif apply_to=='2':
                    apply_to='high'
                elif apply_to=='3':
                    apply_to='low'
                elif apply_to=='4':
                    apply_to='close'

                lleno_forcast = LlenoNNForcast(location=location, csv_file=csv_file,
                                               apply_to=apply_to, function=function, look_back=look_back,
                                               h5model=h5model)
                original_data, dates, prediction, predicted_dates, real_data = lleno_forcast.predict_future(location=location,csv_file=csv_file, apply_to=apply_to,look_back=look_back)
                lleno_forcast.plot_future(original_data=original_data, dates=dates,
                                         prediction=prediction, predicted_dates=predicted_dates,
                                         real_data=real_data)
            elif buildorpredict == 4:
                # Get the average ATP for all 24 hours
                # Get locations
                location = (input('Enter the loction for the lowest timeframe you will be trading on data, enter 0 if data is in this dir...\n')).replace("\\", '\\\\')
                location1 = (input('Enter the loction for the middle timeframe you will be trading on data, enter 0 if data is in this dir...\n')).replace("\\", '\\\\')
                location2 = (input('Enter the loction for the highest timeframe you will be trading on data, enter 0 if data is in this dir...\n')).replace("\\", '\\\\')
                location = f'{location}\\\\'
                location1 = f'{location1}\\\\'
                location2 = f'{location2}\\\\'

                # Get CSV files
                csv_file = input('Enter the name of the file for the lowest timeframe...\n')
                csv_file1 = input('Enter the name of the file for the middle timeframe...\n')
                csv_file2 = input('Enter the name of the file for the highest timeframe...\n')
                

        try:
            quit_ = int(input("Perform another function.\n1. Yes\n2. No\n"))
            if quit_ <= 2:
                if quit_ == 1:
                    pass
                else:
                    keep_looping = False
                    print("GOOD BYE")
            else:
                print("Enter '1' or '2'!!\n GOOD BYE\n")

        except Exception as e:
            # keep_looping = False
            print(e)
            print("Enter a number!!!\nGOOD BYE\n")



# C:\Users\ThembekileMkhombo\Documents\Financial trading\Datasets for machine learning\USDZAR\Training
# USDZAR_Candlestick_1_Hour_BID_04.08.2003-09.04.2022.csv
# USDZAR_Candlestick_4_Hour_BID_04.08.2003-09.04.2022.csv
# USDZAR_Candlestick_1_D_BID_04.08.2003-09.04.2022.csv
# USDZAR_lleno_close.h5


# Fix the 4 hour and 1 day pull extra data  DONE
# allow it to work on any platform          DONE
# Saving history                            DONE
# Allow to make predictions on multiple currencies. TASK FOR NEXT WEEK 
# Check markets like Nasdaq