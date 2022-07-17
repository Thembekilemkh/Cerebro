import pandas as pd
import numpy as np
import requests
from keras.models import load_model
from matplotlib.dates import date2num
from datetime import timedelta, date
from datetime import datetime
import datetime as dt
from tqdm import trange
import pickle
import threading
from queue import Queue
import pytz
import os
import csv
from keras.callbacks import ModelCheckpoint, EarlyStopping
from time import sleep


class MultiTradeSystem():
    def __init__(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        pre_processed_data = kwargs['pre_processed_data']
        system_name = kwargs['system_name']
        
        threading.Thread(target=self.on_start, kwargs={"pre_processed_data":pre_processed_data, "system_name":system_name}).start()
    
    def on_start(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        pre_processed_data = kwargs["pre_processed_data"]
        system_name = kwargs['system_name']
        data_timedelta = self.get_dates(pre_processed_data=pre_processed_data[system_name])
        
        # Get the current day and time 
        self.start_trade_loop(data_timedelta=data_timedelta, pre_processed_data=pre_processed_data, system_name=system_name)

    
    def start_trade_loop(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        # Required variables
        data_timedelta = kwargs["data_timedelta"]
        pre_processed_data=kwargs["pre_processed_data"]
        system_name = kwargs['system_name']
        loop_count = 0


        # THE LOOP BEGINS
        while True:
            # Just delete all files before starting
            self.rename_downloaded_files()

            # Download data required
            data_downloaded = False
            downloaded_data = {}
            for k, v in pre_processed_data[system_name].items():
                downloaded_data_, dd = self.download_data(data_timedelta=data_timedelta, 
                                loop_count=loop_count, frame=k)
                
                downloaded_data[k]=downloaded_data_[k]

                #print(f'dd: {dd}\n key: {k}')
                if dd == True:
                    data_downloaded = True

            if data_downloaded:
                made_pred = ""
                for key, val in downloaded_data.items():
                    made_pred = f"{made_pred}{key}\n"
                    if downloaded_data[key]["downloaded"] == True and downloaded_data[key]["got_live"]:
                        # Decide whether a prediction or a traning is required. Perform each task as per usual
                        new_model, processed_data = self.train_models(downloaded_data=downloaded_data[key], frame=key, pre_processed_data=pre_processed_data[system_name])

                        if new_model:
                            downloaded_data[key]['pre_processed_data'] = processed_data
                            original_data, dates, all_data, predicted_dates, real_data, predicted_dates_num, prediction, current_price, current_date_ = self.make_predictions(downloaded_data=downloaded_data[key], 
                                                                                                                                    model=new_model,
                                                                                                                                    frame=key)
                            """
                            if key == "middle_timeframe":    
                                now = datetime.now()
                                current_date_ = f'{now.year}-{now.month}-{now.day} {now.hour}:00:00'
                                current_date_ = datetime.strptime(current_date_, '%Y-%m-%d %H:%M:%S')"""

                            # Send to Metatrader
                            self.send_coordinates(prediction=prediction, timeframe=key, system_name=system_name)

                            # Save to historical file
                            self.save_history(prediction=prediction, current_date_=current_date_,
                                                timeframe=key, system_name=system_name)

                            # Update variables
                            pre_processed_data[system_name] = self.update_variables(data_timedelta=data_timedelta,
                                                pre_processed_data=pre_processed_data[system_name], downloaded_data=downloaded_data[key],
                                                frame=key)

                        else:
                            pass
                            #print(f'No Model for frame: {key}')

                # Files need to be purged renamed or relaocated
                self.rename_downloaded_files()

                # Save new developments
                self.save_new_pre_processed_data(pre_processed_data=pre_processed_data, system_name=system_name)
                
                print(f"{system_name} made predictions for the following timeframe(s):\n{made_pred}")
            else:
                print("\nNo training or prediction required\n")
            
            # wait to make the next prediction
            #print("\n5 minute wait ...\n")
            sleep(300)

            # Variables for next iteration
            data_timedelta = self.get_dates(pre_processed_data=pre_processed_data)
            loop_count = loop_count+1

    def preprocess(self, **kwargs):
        # Getting training data size
        train_size = int(len(self.data) * 0.8)
 


        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)


        # Creating the train dataset
        train_data = scaled_data[0:train_size, :]


        # Split into the training data and the actual data
        x_train = []
        y_train = []
        look_back = 60
        for i in trange(look_back, len(train_data), desc="Splitting training data"):
            x_train.append(train_data[i - 60:i, :])
            y_train.append(train_data[i, :])


        # Convert x_train and y_train to a numpy array
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Preparing variables for our LSTM
        features = 1
        rows = x_train.shape[0]
        timesteps = x_train.shape[1]
        batch = 1  # 1 2 1459 2918

        # reshape data to a 3D array
        x_train = np.reshape(x_train, (rows, timesteps, features))

        pre_processed_data = {"lowest_timeframe":[x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data],
                              "middle_timeframe":[x_train1, y_train1, batch, train_size1, timesteps1, features, scaler, scaled_data1],
                              "highest_timeframe":[x_train2, y_train2, batch, train_size2, timesteps2, features, scaler, scaled_data2]}
        return pre_processed_data


    def get_dates(self, **kwargs):
        pre_processed_data = kwargs["pre_processed_data"]

        # Get the current day and time
        today = date.today()
        today = today.strftime("%d/%m/%Y")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        tz = pytz.FixedOffset(-120)
        df_now = (now,)#pd.DataFrame({'now': [now]})
        #df_now = pd.to_datetime(df_now['now'].dt.date)

        #print(f'now: {now}')
        #print(f'today: {today}')
        #print(f'df_now: {df_now}')
        #df_now[0] = tz.localize(df_now[0])

        current_year = df_now[0].year
        current_month = df_now[0].month
        current_day = df_now[0].day
        current_hour = df_now[0].hour

        #print(f"Now type: {df_now[0].tzinfo}")
        #print(f"Low type: {pre_processed_data['lowest_timeframe']['last_trained_time'][0].tzinfo}")

        data_timedelta = {}
        get_mid = False
        for key, val in pre_processed_data.items():
            # Required variables
            year_behind = False
            month_behind = False
            day_behind = False
            hour_behind = False
            years = 0
            months = 0
            days = 0
            hours = 0
            
            # Check whether we should download data
            proceed = False
            if key == "lowest_timeframe" and (df_now[0]-timedelta(hours=1) > pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=1)):
                print(f'Lowest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=1)}')
                #print(f'Lowest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]}')
                proceed = True
            
            elif key == "middle_timeframe" and (df_now[0]-timedelta(hours=1) > pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=4)):
                print(f'middle timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=4)}')
                proceed = True
            
            elif key == "highest_timeframe" and (df_now[0] >= pre_processed_data[key]["last_trained_time"][0]+timedelta(days=1)):
                print(f'Highest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]+timedelta(days=1)}')
                #print(f'Highest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]}')
                proceed = True

            data_timedelta[key] = {}
            if proceed:            
                
                # Play around with the date and time to see from where to download data
                diff = df_now[0] - pre_processed_data[key]["last_trained_time"][0]
                #print(f'\nWe are {diff} away from the last recorded time\n')
                #crash = pre_processed_data[key]["Crash"][0]

                # get required variables
                seconds_from_last = diff.seconds
                days_from_last = diff.days

                """print(f"Days from last trained data: {days_from_last}\n
                          Microseconds from last trained data: {diff.microseconds}\n
                          Seconds from last trained data: {seconds_from_last}\n
                          Nano seconds from last trained data: {diff.nanoseconds}\n
                          It's delta man from last trained data: {diff.delta}\n")"""


                if days_from_last >= 1:
                    if days_from_last >= 365:
                        year_behind = True
                        month_behind = True
                        day_behind = True
                        hour_behind = True
                        years = days_from_last/365
                        months = days_from_last/30
                        days = days_from_last
                        hours = seconds_from_last/3600

                    elif days_from_last >= 30:
                        month_behind = True
                        day_behind = True
                        hour_behind = True
                        months = days_from_last/30
                        days = days_from_last
                        hours = seconds_from_last/3600
                    else:
                        day_behind = True
                        hour_behind = True
                        days = days_from_last
                        hours = seconds_from_last/3600

                elif seconds_from_last >= 3600:
                    hours = seconds_from_last/3600
                    hour_behind = True
                else:
                    print("No need to download anything")
                    print(f'We are {diff} away from the last recorded time')

            # Load timedeltas to the dict
            data_timedelta[key]["year_behind"] = year_behind
            data_timedelta[key]["month_behind"] = month_behind
            data_timedelta[key]["day_behind"] = day_behind
            data_timedelta[key]["hour_behind"] = hour_behind
            data_timedelta[key]["current_year"] = current_year
            data_timedelta[key]["current_month"] = current_month
            data_timedelta[key]["current_day"] = current_day
            data_timedelta[key]["current_hour"] = current_hour
            data_timedelta[key]["current_date"] = df_now[0]
            data_timedelta[key]["years"] = years
            data_timedelta[key]["months"] = months
            data_timedelta[key]["days"] = days
            data_timedelta[key]["hours"] = hours

        return data_timedelta

    def get_date(self, **kwargs):
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        date_ = ''

        if month < 10:
            if day < 10:
                date_ = str(year)+"_0"+str(month)+"_0"+str(day)
            else:
                date_ = str(year)+"_0"+str(month)+"_"+str(day)
        else:
            if day < 10:
                date_ = str(year)+"_"+str(month)+"_0"+str(day)
            else:
                date_ = str(year)+"_"+str(month)+"_"+str(day)

        return date_

    def reverse_df(self, **kwargs):
        df = kwargs['df']
        tmp_df = df
        #print(f"DF: {df}")
        last_i = len(df['time']) - 1
        for i in range(len(df['time'])):
            print(f'Index:{last_i} {df["time"].iloc[last_i]}         {i} {df["time"].iloc[i]}')
            for col in df.columns:
                tmp_df[col].iloc[i] = df[col].iloc[last_i]

            last_i = last_i-1
        #print(f"DF: {df}")
        return df
    
    def get_4hour(self, **kwargs):
        df = kwargs['df']

        count = 0
        new_list = []
        for row in range(len(df['time'])):
            date_time_obj = datetime.strptime(df['time'].iloc[row], '%Y-%m-%d %H:%M:%S')
            if (date_time_obj.hour == 0) or (date_time_obj.hour == 4) or (date_time_obj.hour == 8) or (date_time_obj.hour == 12) or (date_time_obj.hour == 16) or (date_time_obj.hour == 20):
                new_list.append((df['time'].iloc[row], df['open'].iloc[row], df['close'].iloc[row], df['high'].iloc[row], df['low'].iloc[row]))
                count = count+1

        df_new = pd.DataFrame (new_list, columns = ['time', 'open', 'close', 'high', 'low'])

        return df_new

    def download(self, **kwargs):
        timeframe = kwargs['timeframe']
        currency_pair = kwargs['currency_pair']
        end_date = kwargs['end_date']
        current_year = kwargs['current_year']
        current_month = kwargs['current_month']
        current_day = kwargs['current_day']
        frame = kwargs['frame']
        data_timedelta = kwargs['data_timedelta']
        key = 'YKKBJH4AG704FDQC'
        from_sym = 'USD'
        to_sym = 'ZAR'
        got_live = False
        data_file = None
        size = 'full'


        if timeframe == 'lowest_timeframe':
            start_date = data_timedelta[frame]["current_date"] - timedelta(hours=(60*3))
            #print(f"Currency: {[currency_pair]}")
            #print(f"Start date: {start_date.date()}\nStart type: {type(start_date.date())}")
            #print(f"Start date: {end_date}\nStart type: {type(end_date)}")
            """
            print('Downloading Duka data...')
            import_ticks([currency_pair], start_date.date(), end_date, 1, TimeFrame.H1, "Currency pair\\", True)
            start_date = self.get_date(year=start_date.year, month=start_date.month, day=start_date.day)
            end_date = self.get_date(year=current_year, month=current_month, day=current_day)

            old_file = "Currency pair\\USDZAR-"+start_date+"-"+end_date+".csv"
            file = "Currency pair\\USDZAR-H1-"+start_date+"-"+end_date+".csv"
            os.rename(old_file, file)
            """
            file = None
            

            try:
                function = 'FX_INTRADAY'
                interval = '60min' # 1min,5min,15min, 30min, 60min
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={from_sym}&to_symbol={to_sym}&interval={interval}&apikey={key}'
                r = requests.get(url)
                data = r.json()

                date = list(data['Time Series FX (60min)'].keys())
                data_file = data['Time Series FX (60min)']

                open_ = []
                high = []
                low = []
                close=[]
                for i in range(len(date)):
                    open_.append(data_file[date[i]]['1. open'])
                    high.append(data_file[date[i]]['2. high'])
                    low.append(data_file[date[i]]['3. low'])
                    close.append(data_file[date[i]]['4. close'])

                de_list = {'time':date[::-1],'open':open_[::-1],'close':close[::-1],
                          'high':high[::-1],'low':low[::-1]}
                df = pd.DataFrame(de_list)#, columns=['time', 'open', 'high', 'low', 'close']
                data_file = df
                got_live = True

            except Exception as e:
                print(f'Failed to download live data: {e}')
                got_live = False
                data_file = {}

            data = [file, data_file, got_live]

        elif timeframe == "middle_timeframe":
            """print('Downloading Duka data...')
            start_date = data_timedelta[frame]["current_date"] - timedelta(hours=(240*3))
            import_ticks([currency_pair], start_date.date(), end_date, 1, TimeFrame.H4, "Currency pair\\", True)
            start_date = self.get_date(year=start_date.year, month=start_date.month, day=start_date.day)
            end_date = self.get_date(year=current_year, month=current_month, day=current_day)

            old_file = "Currency pair\\USDZAR-"+start_date+"-"+end_date+".csv"
            file = "Currency pair\\USDZAR-H4-"+start_date+"-"+end_date+".csv"
            os.rename(old_file, file)
            got_live = False
            data_file = {}"""
            file = None

            try:
                function = 'FX_INTRADAY'
                interval = '60min' # 1min,5min,15min, 30min, 60min
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={from_sym}&to_symbol={to_sym}&interval={interval}&outputsize={size}&apikey={key}'
                r = requests.get(url)
                data = r.json()

                date = list(data['Time Series FX (60min)'].keys())
                data_file = data['Time Series FX (60min)']

                open_ = []
                high = []
                low = []
                close=[]
                for i in range(len(date)):
                    open_.append(data_file[date[i]]['1. open'])
                    high.append(data_file[date[i]]['2. high'])
                    low.append(data_file[date[i]]['3. low'])
                    close.append(data_file[date[i]]['4. close'])

                de_list = {'time':date[::-1],'open':open_[::-1],'close':close[::-1],
                          'high':high[::-1],'low':low[::-1]}
                df = pd.DataFrame(de_list)#, columns=['time', 'open', 'high', 'low', 'close']

                # Get all the four hours
                df = self.get_4hour(df=df)
                data_file = df
                got_live = True

            except Exception as e:
                print(f'Failed to download live data: {e}')
                got_live = False
                data_file = {}

            data = [file, data_file, got_live]

        elif timeframe == "highest_timeframe":
            """
            print('Downloading Duka data...')

            start_date = data_timedelta[frame]["current_date"] - timedelta(days=60*3)
            import_ticks([currency_pair], start_date.date(), end_date, 1, TimeFrame.D1, "Currency pair\\", True)
            start_date = self.get_date(year=start_date.year, month=start_date.month, day=start_date.day)
            end_date = self.get_date(year=current_year, month=current_month, day=current_day)

            old_file = "Currency pair\\USDZAR-"+start_date+"-"+end_date+".csv"
            file = "Currency pair\\USDZAR-D1-"+start_date+"-"+end_date+".csv"
            os.rename(old_file, file)
            """
            file = None
            data_file = {}

            try:
                function = 'FX_DAILY'
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={from_sym}&to_symbol={to_sym}&apikey={key}'
                r = requests.get(url)
                data = r.json()

                date = list(data['Time Series FX (Daily)'].keys())
                data_file = data['Time Series FX (Daily)']

                open_ = []
                high = []
                low = []
                close=[]
                for i in range(len(date)):
                    open_.append(data_file[date[i]]['1. open'])
                    high.append(data_file[date[i]]['2. high'])
                    low.append(data_file[date[i]]['3. low'])
                    close.append(data_file[date[i]]['4. close'])

                de_list = {'time':date[::-1], 'open':open_[::-1],'high':high[::-1],'low':low[::-1],'close':close[::-1]}
                df = pd.DataFrame(de_list)#, columns=['open', 'high', 'low', 'close']
                data_file = df
                got_live = True

            except Exception as e:
                print(f'Failed to download live data: {e}')
                got_live = False
                data = {}

            data = [file, data_file, got_live]

        return data

    def download_data(self, **kwargs):
        data_timedelta = kwargs["data_timedelta"]
        loop_count = kwargs["loop_count"]
        frame = kwargs['frame']
        data_downloaded = True

        #print(f'data timedelta: {data_timedelta}')
        downloaded_data = {}
        if loop_count > -1:
            if data_timedelta[frame]["year_behind"] == False:

                if data_timedelta[frame]["month_behind"] == False:
                    
                    if data_timedelta[frame]["day_behind"] == False:
                        
                        if data_timedelta[frame]["hour_behind"] == False:
                            # no data need to download data at lower levels than this
                            data_downloaded = False
                            downloaded_data[frame]= {}
                            downloaded_data[frame]["downloaded"] = False
                            downloaded_data[frame]["file"] = ''
                            downloaded_data[frame]['train_or_predict'] = ""
                            downloaded_data[frame]['got_live'] = False

                        else:
                            if frame == 'lowest_timeframe' or (frame == 'middle_timeframe' and data_timedelta[frame]["hours"] >= 4):
                                # Prep download data variables
                                downloaded_data[frame]= {}

                                # Just download the hourly data, if we can specify the particular hour from where to start
                                #print(f"\nWe are {data_timedelta[frame]['hours']} hours for the last trained time\n")
                                current_year = data_timedelta[frame]["current_year"]
                                current_month = data_timedelta[frame]["current_month"]
                                current_day = data_timedelta[frame]["current_day"]
                                end_date = dt.date(current_year, current_month, current_day)

                                file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                                    current_year=current_year, current_month=current_month,
                                                     current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                                downloaded_data[frame]["downloaded"] = True
                                downloaded_data[frame]["file"] = file[0]
                                downloaded_data[frame]['train_or_predict'] = "train"
                                downloaded_data[frame]['data'] = file[1]
                                downloaded_data[frame]['got_live'] = file[2]
                            
                    else:
                        # Prep download data variables
                        downloaded_data[frame]= {}

                        # Just download the hourly data, if we can specify the particular hour from where to start
                        #print(f"\nWe are {data_timedelta[frame]['days']} days for the last trained time\n")
                        current_year = data_timedelta[frame]["current_year"]
                        current_month = data_timedelta[frame]["current_month"]
                        current_day = data_timedelta[frame]["current_day"]
                        end_date = dt.date(current_year, current_month, current_day)

                        file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                            current_year=current_year, current_month=current_month, 
                                            current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                        downloaded_data[frame]["downloaded"] = True
                        downloaded_data[frame]["file"] = file[0]
                        downloaded_data[frame]['train_or_predict'] = "train"
                        downloaded_data[frame]['data'] = file[1]
                        downloaded_data[frame]['got_live'] = file[2]
                        

                else:
                    # Prep download data variables
                    downloaded_data[frame]= {}

                    # Just download the hourly data, if we can specify the particular hour from where to start
                    #print(f"\nWe are {data_timedelta[frame]['months']} months for the last trained time\n")
                    current_year = data_timedelta[frame]["current_year"]
                    current_month = data_timedelta[frame]["current_month"]
                    current_day = data_timedelta[frame]["current_day"]
                    end_date = dt.date(current_year, current_month, current_day)

                    file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                        current_year=current_year, current_month=current_month,
                                         current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                    downloaded_data[frame]["downloaded"] = True
                    downloaded_data[frame]["file"] = file[0]
                    downloaded_data[frame]['train_or_predict'] = "train"
                    downloaded_data[frame]["data"] = file[1]
                    downloaded_data[frame]['got_live'] = file[2]
            else:
                # Prep download data variables
                downloaded_data[frame]= {}

                # Just download the hourly data, if we can specify the particular hour from where to start
                #print(f"\nWe are {data_timedelta[frame]['years']} years for the last trained time\n")
                current_year = data_timedelta[frame]["current_year"]
                current_month = data_timedelta[frame]["current_month"]
                current_day = data_timedelta[frame]["current_day"]
                end_date = dt.date(current_year, current_month, current_day)

                file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                    current_year=current_year, current_month=current_month, 
                                    current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                downloaded_data[frame]["downloaded"] = True
                downloaded_data[frame]["file"] = file[0]
                downloaded_data[frame]['train_or_predict'] = "train"
                downloaded_data[frame]['data'] = file[1]
                downloaded_data[frame]['got_live'] = file[2]

        else:
            # We come through here if it's not the first loop. What you see is that it is very similar to the first loop seeminly useless
            # We actually better not come through here I ain't got time for this
            pass
        
        return downloaded_data, data_downloaded

    def preprocess_single_batch(self, **kwargs):
        # Get required data
        data_file = kwargs['data']
        got_live = kwargs['got_live']
        df_ready = kwargs['df_ready']
        pre_processed_data = kwargs['pre_processed_data']
        timeframe = kwargs['timeframe']
        frame = kwargs['frame']
        c_pair = kwargs["c_pair"]

        # Unpack data file
        if got_live:
            tikData = df_ready
            tikData.columns = [["time", "open", "close", "high", "low"]]

        else:
            tikData = pd.read_csv(data_file, parse_dates=[0])
            tikData.columns = [["time", "open", "close", "high", "low"]]

        #print(got_live)
        #print(f'Live columns: {tikData.columns}')
        #print(f'Downloaded columns: {tikData1.columns}')
        timeData2_ = tikData[['time']]
        timeData2 = timeData2_.to_numpy()
        last_timestamp = timeData2_.iloc[-1]
        data_ = tikData[['close']]
        data = data_.to_numpy()
        new_last_close = data_.iloc[-1]


        if frame == 'lowest_timeframe':
            if isinstance(last_timestamp[0], str):
                last_timestamp[0]=datetime.strptime(last_timestamp[0], '%Y-%m-%d %H:%M:%S')
            next_close_time = last_timestamp[0]+timedelta(hours=1)
        elif frame == 'middle_timeframe':
            if isinstance(last_timestamp[0], str):
                last_timestamp[0]=datetime.strptime(last_timestamp[0], '%Y-%m-%d %H:%M:%S')
            next_close_time = last_timestamp[0]+timedelta(hours=4)
        elif frame == 'highest_timeframe':
            if isinstance(last_timestamp[0], str):
                time_ = "00:00:00"
                last_timestamp[0]=datetime.strptime(f'{last_timestamp[0]} {time_}', '%Y-%m-%d %H:%M:%S')
            next_close_time = last_timestamp[0]+timedelta(days=1)

        # Getting training data size
        train_size = int(len(data) * 0.8)


        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)


        # Split into the training data and the actual data
        x_train = []
        y_train = []
        look_back = 60
        for i in trange(look_back, len(scaled_data), desc="Splitting training data"):
            x_train.append(scaled_data[i - 60:i, :])
            y_train.append(scaled_data[i, :])

        # Convert x_train and y_train to a numpy array
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Preparing variables for our LSTM
        #print(f'x train: {x_train}')
        #print(f'Train shape: {x_train.shape}')
        features = 1
        rows = x_train.shape[0]
        timesteps = x_train.shape[1]
        batch = 1  # 1 2 1459 2918

        # reshape data to a 3D array
        x_train = np.reshape(x_train, (rows, timesteps, features))

        pre_processed_data = {"x_train":x_train, "y_train":y_train, "batch":batch, 
                              "train_size":train_size, "timesteps":timesteps, "features":features, 
                              "scaler":scaler, "scaled_data": scaled_data, "look_back":look_back,
                              "last_trained_time":last_timestamp, 'last_close':new_last_close,
                              "next_close_time":next_close_time, "timeframe":timeframe, 
                              "c_pair":c_pair}
        return pre_processed_data

    def make_predictions(self, **kwargs):
        downloaded_data = kwargs['downloaded_data']
        model = kwargs['model']
        frame = kwargs['frame']

        if downloaded_data['train_or_predict'] == "predict" or downloaded_data['train_or_predict'] == "train":
            csv_file = downloaded_data["file"]
            look_back = downloaded_data['pre_processed_data']['look_back']

            # Get data
            if downloaded_data['got_live']:
                tikData = downloaded_data['data']
                tikData.columns = [["time", "open", "close", "high", "low"]]
            else:
                d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                tikData = pd.read_csv(csv_file, parse_dates=[0], date_parser=d_parser, delimiter=',')
                tikData.columns = [["time", "open", "close", "high", "low"]]

            data = tikData[["close"]]
            data_ = tikData[["close"]]
            #data.drop_duplicates(subset ="date", keep=False, inplace=True)

            # Date preprocessing
            date_ = tikData[["time"]]  # Date type:  <class 'pandas._libs.tslibs.timestamps.Timestamp'>
            date = date_.to_numpy()
            # date = date2num(date)

            # Get the last data the length of the look back period
            last_look_back = data[-look_back:]
            date = date[-look_back:]
            data = last_look_back.to_numpy()

            # Scale data between 0 and 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Create an empty list
            X_data = []
            predicted_data = []
            predicted_data_ = []
            predicted_dates = []
            all_data = []
            all_dates = []

            # Append last look back data
            X_data.append(scaled_data)

            # Prepare a list that will contain all the data
            for y in range(len(scaled_data)):
                all_data.append(scaled_data[y][0])
                all_dates.append(date[y][0])#.to_pydatetime()

            # Convert to numpy
            X_data = np.array(X_data)
            dates = np.array(all_dates)
            original_data = X_data
            current_date = date_.iloc[-1]
            current_date = current_date[0]
            current_date_ = current_date
            current_price = data_.iloc[-1][0]


            # Loop through and keep getting new predicted data
            x = 0
            while x < 1:
                # Reshape to 3D Array
                X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

                # Get predicted scaled price
                predicted_price = model.predict(X_data)

                # Increment dates
                if frame == 'lowest_tiemframe':
                    current_date = current_date + timedelta(hours=1)
                elif frame == 'middle_timeframe':
                    if isinstance(current_date, str):
                        current_date = datetime.strptime(f'{current_date}', '%Y-%m-%d %H:%M:%S')
                    current_date = current_date + timedelta(hours=4)

                elif frame == 'highest_timeframe':
                    if isinstance(current_date, str):
                        current_date = datetime.strptime(f'{current_date} 00:00:00', '%Y-%m-%d %H:%M:%S')
                    current_date = current_date + timedelta(days=1)


                # Append predicted data where needed
                X_data = np.append(X_data, predicted_price[0][0])
                predicted_data.append(predicted_price[0][0])
                predicted_data_.append(predicted_price[0][0])
                all_data.append(predicted_price[0][0])
                predicted_dates.append(current_date)

                # Get the last look back and prepare data for next round
                X_data = X_data[-look_back:]

                X_data = np.reshape(X_data, (1, X_data.shape[0], 1))

                # Add one hour to the date array

                x = x + 1

            #print(f"\nData shape: {X_data.shape}")
            #print(f"Our predicted date: {predicted_dates[0]}")
            X_data = np.reshape(X_data, (X_data.shape[1]))

            '''
                predictions = scaler.inverse_transform(predictions)
                predict_df = pd.DataFrame(data=predictions, columns=["Predictions"])


                mse = np.sqrt(np.mean(predictions - y_test) ** 2)'''

            # Turn to numpy and shape data to be reshaped
            prediction = all_data[-121:]
            prediction = np.array(prediction)
            prediction = np.reshape(prediction, (prediction.shape[0], 1))
            num_all = np.array(all_data)
            num_all = np.reshape(num_all, (num_all.shape[0], 1))
            original_data = np.reshape(original_data, (original_data.shape[1], 1))
            predicted_dates = np.array(predicted_dates)
            predicted_dates_num = date2num(predicted_dates)
            predicted_data = np.array(predicted_data)
            predicted_data = np.reshape(predicted_data, (predicted_data.shape[0], 1))


            # Inverse transform
            real_data = scaler.inverse_transform(num_all)
            original_data = scaler.inverse_transform(original_data)
            all_data = scaler.inverse_transform(prediction)
            predicted_data = scaler.inverse_transform(predicted_data)



        else:
            print('\n No prediction made\n')

        return original_data, dates, all_data, predicted_dates, real_data, predicted_dates_num, predicted_data, current_price, current_date_

    def train_model(self, **kwargs):
        # Get relavent data
        pre_processed_data=kwargs['pre_processed_data']
        model = kwargs['model']

        x_train = pre_processed_data['x_train']
        y_train = pre_processed_data['y_train']
        batch = pre_processed_data['batch']
        timeframe=pre_processed_data['timeframe']
        c_pair = pre_processed_data['c_pair']

        # Saves the model every now and again incase something happens
        checkpoint = ModelCheckpoint('', save_best_only=True)

        # Getting our model ready to stop early if it stops improving
        ES = EarlyStopping()

        # Load existing model
        model = load_model(model)

        # Train the model
        model.fit(x_train, y_train,
                  batch_size=batch, epochs=1,
                  callbacks=[checkpoint, ES])  # shuffle=False
        # model.reset_states()

        # Saving the model for later use
        #print("Model created")
        model.save(c_pair + f"_lleno_close_{timeframe}.h5")

        return model



    def train_models(self, **kwargs):

        # figure out which model need to be trained
        downloaded_data = kwargs["downloaded_data"]
        frame = kwargs['frame']
        processed_data=kwargs['pre_processed_data']

        #print(f"""Train or Predict:{downloaded_data["train_or_predict"]}\nframe: {frame}""")
        if (downloaded_data["train_or_predict"] == "train" or downloaded_data["train_or_predict"] == "predict") and frame=="lowest_timeframe":
            timeframe = '1Hour'
            c_pair = "USDZAR"
            processed_data = self.preprocess_single_batch(data=downloaded_data['file'], pre_processed_data=processed_data, 
                                                          timeframe=timeframe, c_pair=c_pair, frame=frame, 
                                                          got_live=downloaded_data['got_live'], df_ready=downloaded_data['data'])
            new_model = self.train_model(model="USDZAR_lleno_close_1Hour.h5", pre_processed_data=processed_data)
                       
        elif (downloaded_data["train_or_predict"] == "train" or downloaded_data["train_or_predict"] == "predict") and frame=="middle_timeframe":
            timeframe = '4Hour'
            c_pair = "USDZAR"
            processed_data = self.preprocess_single_batch(data=downloaded_data['file'], pre_processed_data=processed_data, 
                                                          timeframe=timeframe, c_pair=c_pair, frame=frame, 
                                                          got_live=downloaded_data['got_live'], df_ready=downloaded_data['data'])
            new_model = self.train_model(model="USDZAR_lleno_close_4hour.h5", pre_processed_data=processed_data)
            

        elif (downloaded_data["train_or_predict"] == "train" or downloaded_data["train_or_predict"] == "predict") and frame=="highest_timeframe":
            timeframe = '1D'
            c_pair = "USDZAR"
            processed_data = self.preprocess_single_batch(data=downloaded_data['file'], pre_processed_data=processed_data, 
                                                          timeframe=timeframe, c_pair=c_pair, frame=frame, 
                                                          got_live=downloaded_data['got_live'], df_ready=downloaded_data['data'])
            new_model = self.train_model(model="USDZAR_lleno_close_1D.h5", pre_processed_data=processed_data)

        else:
            new_model = False
            processed_data = False

        # Record new last trained dates
        return new_model, processed_data
    
    def find_paths(self, **kwargs):
        documents = True
        metatrader = True

        rootdir = "C:\\Users\\"
        to_meta = "AppData\\Roaming\\MetaQuotes\\Terminal"
        more_meta = "MQL5\\Files"
        to_docs = "Documents\\Financial trading\\Datasets for machine learning\\Predictions made"
        paths = []

        for file in os.listdir(rootdir):
            d = f"{rootdir}{file}"
            if os.path.isdir(d):
                if os.path.exists(f"{d}\\{to_docs}"):
                    if documents:
                        paths.append(f"{d}\\{to_docs}")
                if os.path.exists(f"{d}\\{to_meta}"):
                    for f in os.listdir(f"{d}\\{to_meta}"):
                        dir = f"{d}\\{to_meta}\\{f}"
                        if os.path.isdir(dir):
                            if os.path.exists(f"{dir}\\{more_meta}"):
                                if metatrader:
                                    paths.append(f"{dir}\\{more_meta}")
  
        return paths
  
  
    def save_history(self, **kwargs):
        prediction = kwargs["prediction"]
        timeframe = kwargs["timeframe"]
        current_date_ = kwargs["current_date_"]
        system_name = kwargs[system_name]
        filename = f"Historical data {timeframe}"
        paths = self.find_paths()
        
        # Choose where is save file according to the OS
        for path in paths:
            with open(f'{path}\\{system_name} {filename}.txt', 'a') as f:
                for i in range(len(prediction)):
                    f.write(f"{current_date_},{str(prediction[i][0])}\n")

             
    def send_coordinates(self, **kwargs):
        prediction = kwargs["prediction"]
        timeframe = kwargs["timeframe"]
        system_name = kwargs['system_name']
        paths = self.find_paths()

        # Choose where is save file according to the OS
        for path in paths:
            with open(f'{path}\\{system_name} {timeframe}.txt', 'w') as f:
                for i in range(len(prediction)):
                    f.write(str(prediction[i][0]))

            with open(f'{path}\\{system_name} {timeframe}.csv', mode='w') as file:
                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                for i in range(len(prediction)):
                    #file_writer.writerow([current_date_, current_price, predicted_dates[i], prediction[i][0]])
                    file_writer.writerow([prediction[i][0]])

                print('\nNew prediction saved and sent to Metatrader!!\n')

        
    
    def update_variables(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        downloaded_data = kwargs['downloaded_data']
        frame = kwargs['frame']
        pre_processed_data = kwargs["pre_processed_data"]
        data_timedelta = kwargs['data_timedelta']

        """"x_train":x_train, "y_train":y_train, "batch":batch,
        "train_size":train_size, "timesteps":timesteps, "features":features, 
        "scaler":scaler, "scaled_data": scaled_data, "look_back":look_back,
        "last_trained_time":last_timestamp, 'last_close':new_last_close,
        "next_close_time":next_close_time"""


        pre_processed_data[frame]['x_train'] = downloaded_data["pre_processed_data"]["x_train"]
        pre_processed_data[frame]['y_train'] = downloaded_data["pre_processed_data"]["y_train"]
        pre_processed_data[frame]['batch'] = downloaded_data["pre_processed_data"]["batch"]

        pre_processed_data[frame]['train_size'] = downloaded_data["pre_processed_data"]["train_size"]
        pre_processed_data[frame]['timesteps'] = downloaded_data["pre_processed_data"]["timesteps"]
        pre_processed_data[frame]['features'] = downloaded_data["pre_processed_data"]["features"]

        pre_processed_data[frame]['scaler'] = downloaded_data["pre_processed_data"]["scaler"]
        pre_processed_data[frame]['scaled_data'] = downloaded_data["pre_processed_data"]["scaled_data"]
        pre_processed_data[frame]['look_back'] = downloaded_data["pre_processed_data"]["look_back"]

        pre_processed_data[frame]['last_trained_time'] = downloaded_data["pre_processed_data"]["last_trained_time"]
        pre_processed_data[frame]['last_close'] = downloaded_data["pre_processed_data"]["last_close"]
        pre_processed_data[frame]['next_close_time'] = downloaded_data["pre_processed_data"]["next_close_time"]


        pre_processed_data[frame]['c_pair'] = downloaded_data["pre_processed_data"]["c_pair"]
        pre_processed_data[frame]['timeframe'] = downloaded_data["pre_processed_data"]["timeframe"]

        return pre_processed_data

    def save_new_pre_processed_data(self, **kwargs):
        pre_processed_data = kwargs['pre_processed_data']
        system_name = kwargs['system_name']

        with open(f'{system_name} trained_history.pickle', "wb") as f:
            pickle.dump(pre_processed_data, f)


    def rename_downloaded_files(self, **kwargs):
        
        for file in os.listdir('Currency pair//'):
            if file[-1] == "v":
                if os.path.exists('Currency pair//'+file):
                    os.remove('Currency pair//'+file)
                else:
                    print("The file does not exist")
        

class TradeSystem():
    def __init__(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        pre_processed_data = kwargs['pre_processed_data']

        self.on_start(pre_processed_data=pre_processed_data)

    def on_start(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        pre_processed_data = kwargs["pre_processed_data"]
        data_timedelta = self.get_dates(pre_processed_data=pre_processed_data)
        
        # Get the current day and time 
        self.start_trade_loop(data_timedelta=data_timedelta, pre_processed_data=pre_processed_data)

    def preprocess(self, **kwargs):
        # Getting training data size
        train_size = int(len(self.data) * 0.8)
 


        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data)


        # Creating the train dataset
        train_data = scaled_data[0:train_size, :]


        # Split into the training data and the actual data
        x_train = []
        y_train = []
        look_back = 60
        for i in trange(look_back, len(train_data), desc="Splitting training data"):
            x_train.append(train_data[i - 60:i, :])
            y_train.append(train_data[i, :])


        # Convert x_train and y_train to a numpy array
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Preparing variables for our LSTM
        features = 1
        rows = x_train.shape[0]
        timesteps = x_train.shape[1]
        batch = 1  # 1 2 1459 2918

        # reshape data to a 3D array
        x_train = np.reshape(x_train, (rows, timesteps, features))

        pre_processed_data = {"lowest_timeframe":[x_train, y_train, batch, train_size, timesteps, features, scaler, scaled_data],
                              "middle_timeframe":[x_train1, y_train1, batch, train_size1, timesteps1, features, scaler, scaled_data1],
                              "highest_timeframe":[x_train2, y_train2, batch, train_size2, timesteps2, features, scaler, scaled_data2]}
        return pre_processed_data


    def get_dates(self, **kwargs):
        pre_processed_data = kwargs["pre_processed_data"]

        # Get the current day and time
        today = date.today()
        today = today.strftime("%d/%m/%Y")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        tz = pytz.FixedOffset(-120)
        df_now = (now,)#pd.DataFrame({'now': [now]})
        #df_now = pd.to_datetime(df_now['now'].dt.date)

        #print(f'now: {now}')
        #print(f'today: {today}')
        #print(f'df_now: {df_now}')
        #df_now[0] = tz.localize(df_now[0])

        current_year = df_now[0].year
        current_month = df_now[0].month
        current_day = df_now[0].day
        current_hour = df_now[0].hour

        #print(f"Now type: {df_now[0].tzinfo}")
        #print(f"Low type: {pre_processed_data['lowest_timeframe']['last_trained_time'][0].tzinfo}")

        data_timedelta = {}
        get_mid = False
        for key, val in pre_processed_data.items():
            # Required variables
            year_behind = False
            month_behind = False
            day_behind = False
            hour_behind = False
            years = 0
            months = 0
            days = 0
            hours = 0
            
            # Check whether we should download data
            proceed = False
            if key == "lowest_timeframe" and (df_now[0]-timedelta(hours=1) > pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=1)):
                print(f'Lowest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=1)}')
                #print(f'Lowest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]}')
                proceed = True
            
            elif key == "middle_timeframe" and (df_now[0]-timedelta(hours=1) > pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=4)):
                print(f'middle timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]+timedelta(hours=4)}')
                proceed = True
            
            elif key == "highest_timeframe" and (df_now[0] >= pre_processed_data[key]["last_trained_time"][0]+timedelta(days=1)):
                print(f'Highest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]+timedelta(days=1)}')
                #print(f'Highest timeframe: {df_now[0]} > {pre_processed_data[key]["last_trained_time"][0]}')
                proceed = True

            data_timedelta[key] = {}
            if proceed:            
                
                # Play around with the date and time to see from where to download data
                diff = df_now[0] - pre_processed_data[key]["last_trained_time"][0]
                #print(f'\nWe are {diff} away from the last recorded time\n')
                #crash = pre_processed_data[key]["Crash"][0]

                # get required variables
                seconds_from_last = diff.seconds
                days_from_last = diff.days

                """print(f"Days from last trained data: {days_from_last}\n
                          Microseconds from last trained data: {diff.microseconds}\n
                          Seconds from last trained data: {seconds_from_last}\n
                          Nano seconds from last trained data: {diff.nanoseconds}\n
                          It's delta man from last trained data: {diff.delta}\n")"""


                if days_from_last >= 1:
                    if days_from_last >= 365:
                        year_behind = True
                        month_behind = True
                        day_behind = True
                        hour_behind = True
                        years = days_from_last/365
                        months = days_from_last/30
                        days = days_from_last
                        hours = seconds_from_last/3600

                    elif days_from_last >= 30:
                        month_behind = True
                        day_behind = True
                        hour_behind = True
                        months = days_from_last/30
                        days = days_from_last
                        hours = seconds_from_last/3600
                    else:
                        day_behind = True
                        hour_behind = True
                        days = days_from_last
                        hours = seconds_from_last/3600

                elif seconds_from_last >= 3600:
                    hours = seconds_from_last/3600
                    hour_behind = True
                else:
                    print("No need to download anything")
                    print(f'We are {diff} away from the last recorded time')

            # Load timedeltas to the dict
            data_timedelta[key]["year_behind"] = year_behind
            data_timedelta[key]["month_behind"] = month_behind
            data_timedelta[key]["day_behind"] = day_behind
            data_timedelta[key]["hour_behind"] = hour_behind
            data_timedelta[key]["current_year"] = current_year
            data_timedelta[key]["current_month"] = current_month
            data_timedelta[key]["current_day"] = current_day
            data_timedelta[key]["current_hour"] = current_hour
            data_timedelta[key]["current_date"] = df_now[0]
            data_timedelta[key]["years"] = years
            data_timedelta[key]["months"] = months
            data_timedelta[key]["days"] = days
            data_timedelta[key]["hours"] = hours

        return data_timedelta

    def get_date(self, **kwargs):
        year = kwargs['year']
        month = kwargs['month']
        day = kwargs['day']
        date_ = ''

        if month < 10:
            if day < 10:
                date_ = str(year)+"_0"+str(month)+"_0"+str(day)
            else:
                date_ = str(year)+"_0"+str(month)+"_"+str(day)
        else:
            if day < 10:
                date_ = str(year)+"_"+str(month)+"_0"+str(day)
            else:
                date_ = str(year)+"_"+str(month)+"_"+str(day)

        return date_

    def reverse_df(self, **kwargs):
        df = kwargs['df']
        tmp_df = df
        #print(f"DF: {df}")
        last_i = len(df['time']) - 1
        for i in range(len(df['time'])):
            print(f'Index:{last_i} {df["time"].iloc[last_i]}         {i} {df["time"].iloc[i]}')
            for col in df.columns:
                tmp_df[col].iloc[i] = df[col].iloc[last_i]

            last_i = last_i-1
        #print(f"DF: {df}")
        return df
    
    def get_4hour(self, **kwargs):
        df = kwargs['df']

        count = 0
        new_list = []
        for row in range(len(df['time'])):
            date_time_obj = datetime.strptime(df['time'].iloc[row], '%Y-%m-%d %H:%M:%S')
            if (date_time_obj.hour == 0) or (date_time_obj.hour == 4) or (date_time_obj.hour == 8) or (date_time_obj.hour == 12) or (date_time_obj.hour == 16) or (date_time_obj.hour == 20):
                new_list.append((df['time'].iloc[row], df['open'].iloc[row], df['close'].iloc[row], df['high'].iloc[row], df['low'].iloc[row]))
                count = count+1

        df_new = pd.DataFrame (new_list, columns = ['time', 'open', 'close', 'high', 'low'])

        return df_new

    def download(self, **kwargs):
        timeframe = kwargs['timeframe']
        currency_pair = kwargs['currency_pair']
        end_date = kwargs['end_date']
        current_year = kwargs['current_year']
        current_month = kwargs['current_month']
        current_day = kwargs['current_day']
        frame = kwargs['frame']
        data_timedelta = kwargs['data_timedelta']
        key = 'YKKBJH4AG704FDQC'
        from_sym = 'USD'
        to_sym = 'ZAR'
        got_live = False
        data_file = None
        size = 'full'


        if timeframe == 'lowest_timeframe':
            start_date = data_timedelta[frame]["current_date"] - timedelta(hours=(60*3))
            #print(f"Currency: {[currency_pair]}")
            #print(f"Start date: {start_date.date()}\nStart type: {type(start_date.date())}")
            #print(f"Start date: {end_date}\nStart type: {type(end_date)}")
            """
            print('Downloading Duka data...')
            import_ticks([currency_pair], start_date.date(), end_date, 1, TimeFrame.H1, "Currency pair\\", True)
            start_date = self.get_date(year=start_date.year, month=start_date.month, day=start_date.day)
            end_date = self.get_date(year=current_year, month=current_month, day=current_day)

            old_file = "Currency pair\\USDZAR-"+start_date+"-"+end_date+".csv"
            file = "Currency pair\\USDZAR-H1-"+start_date+"-"+end_date+".csv"
            os.rename(old_file, file)
            """
            file = None
            

            try:
                print("Downloading live data...")
                function = 'FX_INTRADAY'
                interval = '60min' # 1min,5min,15min, 30min, 60min
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={from_sym}&to_symbol={to_sym}&interval={interval}&apikey={key}'
                r = requests.get(url)
                data = r.json()

                date = list(data['Time Series FX (60min)'].keys())
                data_file = data['Time Series FX (60min)']

                open_ = []
                high = []
                low = []
                close=[]
                for i in range(len(date)):
                    open_.append(data_file[date[i]]['1. open'])
                    high.append(data_file[date[i]]['2. high'])
                    low.append(data_file[date[i]]['3. low'])
                    close.append(data_file[date[i]]['4. close'])

                de_list = {'time':date[::-1],'open':open_[::-1],'close':close[::-1],
                          'high':high[::-1],'low':low[::-1]}
                df = pd.DataFrame(de_list)#, columns=['time', 'open', 'high', 'low', 'close']
                data_file = df
                got_live = True

            except Exception as e:
                print(f'Failed to download live data: {e}')
                got_live = False
                data_file = {}

            data = [file, data_file, got_live]

        elif timeframe == "middle_timeframe":
            """print('Downloading Duka data...')
            start_date = data_timedelta[frame]["current_date"] - timedelta(hours=(240*3))
            import_ticks([currency_pair], start_date.date(), end_date, 1, TimeFrame.H4, "Currency pair\\", True)
            start_date = self.get_date(year=start_date.year, month=start_date.month, day=start_date.day)
            end_date = self.get_date(year=current_year, month=current_month, day=current_day)

            old_file = "Currency pair\\USDZAR-"+start_date+"-"+end_date+".csv"
            file = "Currency pair\\USDZAR-H4-"+start_date+"-"+end_date+".csv"
            os.rename(old_file, file)
            got_live = False
            data_file = {}"""
            file = None

            try:
                function = 'FX_INTRADAY'
                interval = '60min' # 1min,5min,15min, 30min, 60min
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={from_sym}&to_symbol={to_sym}&interval={interval}&outputsize={size}&apikey={key}'
                r = requests.get(url)
                data = r.json()

                date = list(data['Time Series FX (60min)'].keys())
                data_file = data['Time Series FX (60min)']

                open_ = []
                high = []
                low = []
                close=[]
                for i in range(len(date)):
                    open_.append(data_file[date[i]]['1. open'])
                    high.append(data_file[date[i]]['2. high'])
                    low.append(data_file[date[i]]['3. low'])
                    close.append(data_file[date[i]]['4. close'])

                de_list = {'time':date[::-1],'open':open_[::-1],'close':close[::-1],
                          'high':high[::-1],'low':low[::-1]}
                df = pd.DataFrame(de_list)#, columns=['time', 'open', 'high', 'low', 'close']

                # Get all the four hours
                df = self.get_4hour(df=df)
                data_file = df
                got_live = True

            except Exception as e:
                print(f'Failed to download live data: {e}')
                got_live = False
                data_file = {}

            data = [file, data_file, got_live]

        elif timeframe == "highest_timeframe":
            """
            print('Downloading Duka data...')

            start_date = data_timedelta[frame]["current_date"] - timedelta(days=60*3)
            import_ticks([currency_pair], start_date.date(), end_date, 1, TimeFrame.D1, "Currency pair\\", True)
            start_date = self.get_date(year=start_date.year, month=start_date.month, day=start_date.day)
            end_date = self.get_date(year=current_year, month=current_month, day=current_day)

            old_file = "Currency pair\\USDZAR-"+start_date+"-"+end_date+".csv"
            file = "Currency pair\\USDZAR-D1-"+start_date+"-"+end_date+".csv"
            os.rename(old_file, file)
            """
            file = None
            data_file = {}

            try:
                print('Downloading live data...')
                function = 'FX_DAILY'
                url = f'https://www.alphavantage.co/query?function={function}&from_symbol={from_sym}&to_symbol={to_sym}&apikey={key}'
                r = requests.get(url)
                data = r.json()

                date = list(data['Time Series FX (Daily)'].keys())
                data_file = data['Time Series FX (Daily)']

                open_ = []
                high = []
                low = []
                close=[]
                for i in range(len(date)):
                    open_.append(data_file[date[i]]['1. open'])
                    high.append(data_file[date[i]]['2. high'])
                    low.append(data_file[date[i]]['3. low'])
                    close.append(data_file[date[i]]['4. close'])

                de_list = {'time':date[::-1], 'open':open_[::-1],'high':high[::-1],'low':low[::-1],'close':close[::-1]}
                df = pd.DataFrame(de_list)#, columns=['open', 'high', 'low', 'close']
                data_file = df
                got_live = True

            except Exception as e:
                print(f'Failed to download live data: {e}')
                got_live = False
                data = {}

            data = [file, data_file, got_live]

        return data

    def download_data(self, **kwargs):
        pre_processed_data = kwargs["pre_processed_data"]
        data_timedelta = kwargs["data_timedelta"]
        loop_count = kwargs["loop_count"]
        frame = kwargs['frame']
        data_downloaded = True

        #print(f'data timedelta: {data_timedelta}')
        downloaded_data = {}
        if loop_count > -1:
            if data_timedelta[frame]["year_behind"] == False:

                if data_timedelta[frame]["month_behind"] == False:
                    
                    if data_timedelta[frame]["day_behind"] == False:
                        
                        if data_timedelta[frame]["hour_behind"] == False:
                            # no data need to download data at lower levels than this
                            data_downloaded = False
                            downloaded_data[frame]= {}
                            downloaded_data[frame]["downloaded"] = False
                            downloaded_data[frame]["file"] = ''
                            downloaded_data[frame]['train_or_predict'] = ""
                            downloaded_data[frame]['got_live'] = False

                        else:
                            if frame == 'lowest_timeframe' or (frame == 'middle_timeframe' and data_timedelta[frame]["hours"] >= 4):
                                # Prep download data variables
                                downloaded_data[frame]= {}

                                # Just download the hourly data, if we can specify the particular hour from where to start
                                #print(f"\nWe are {data_timedelta[frame]['hours']} hours for the last trained time\n")
                                current_year = data_timedelta[frame]["current_year"]
                                current_month = data_timedelta[frame]["current_month"]
                                current_day = data_timedelta[frame]["current_day"]
                                end_date = dt.date(current_year, current_month, current_day)

                                file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                                    current_year=current_year, current_month=current_month,
                                                     current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                                downloaded_data[frame]["downloaded"] = True
                                downloaded_data[frame]["file"] = file[0]
                                downloaded_data[frame]['train_or_predict'] = "train"
                                downloaded_data[frame]['data'] = file[1]
                                downloaded_data[frame]['got_live'] = file[2]
                            
                    else:
                        # Prep download data variables
                        downloaded_data[frame]= {}

                        # Just download the hourly data, if we can specify the particular hour from where to start
                        #print(f"\nWe are {data_timedelta[frame]['days']} days for the last trained time\n")
                        current_year = data_timedelta[frame]["current_year"]
                        current_month = data_timedelta[frame]["current_month"]
                        current_day = data_timedelta[frame]["current_day"]
                        end_date = dt.date(current_year, current_month, current_day)

                        file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                            current_year=current_year, current_month=current_month, 
                                            current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                        downloaded_data[frame]["downloaded"] = True
                        downloaded_data[frame]["file"] = file[0]
                        downloaded_data[frame]['train_or_predict'] = "train"
                        downloaded_data[frame]['data'] = file[1]
                        downloaded_data[frame]['got_live'] = file[2]
                        

                else:
                    # Prep download data variables
                    downloaded_data[frame]= {}

                    # Just download the hourly data, if we can specify the particular hour from where to start
                    #print(f"\nWe are {data_timedelta[frame]['months']} months for the last trained time\n")
                    current_year = data_timedelta[frame]["current_year"]
                    current_month = data_timedelta[frame]["current_month"]
                    current_day = data_timedelta[frame]["current_day"]
                    end_date = dt.date(current_year, current_month, current_day)

                    file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                        current_year=current_year, current_month=current_month,
                                         current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                    downloaded_data[frame]["downloaded"] = True
                    downloaded_data[frame]["file"] = file[0]
                    downloaded_data[frame]['train_or_predict'] = "train"
                    downloaded_data[frame]["data"] = file[1]
                    downloaded_data[frame]['got_live'] = file[2]
            else:
                # Prep download data variables
                downloaded_data[frame]= {}

                # Just download the hourly data, if we can specify the particular hour from where to start
                #print(f"\nWe are {data_timedelta[frame]['years']} years for the last trained time\n")
                current_year = data_timedelta[frame]["current_year"]
                current_month = data_timedelta[frame]["current_month"]
                current_day = data_timedelta[frame]["current_day"]
                end_date = dt.date(current_year, current_month, current_day)

                file = self.download(timeframe=frame, currency_pair="USDZAR",end_date=end_date,
                                    current_year=current_year, current_month=current_month, 
                                    current_day=current_day, frame=frame, data_timedelta=data_timedelta)

                downloaded_data[frame]["downloaded"] = True
                downloaded_data[frame]["file"] = file[0]
                downloaded_data[frame]['train_or_predict'] = "train"
                downloaded_data[frame]['data'] = file[1]
                downloaded_data[frame]['got_live'] = file[2]

        else:
            # We come through here if it's not the first loop. What you see is that it is very similar to the first loop seeminly useless
            # We actually better not come through here I ain't got time for this
            pass
        
        return downloaded_data, data_downloaded

    def preprocess_single_batch(self, **kwargs):
        # Get required data
        data_file = kwargs['data']
        got_live = kwargs['got_live']
        df_ready = kwargs['df_ready']
        pre_processed_data = kwargs['pre_processed_data']
        timeframe = kwargs['timeframe']
        frame = kwargs['frame']
        c_pair = kwargs["c_pair"]

        # Unpack data file
        print(f'Frame: {frame}')
        print(f'\ndf_ready: {df_ready}\n')
        print(f'got_live: {got_live}')
        if got_live:
            tikData = df_ready
            tikData.columns = [["time", "open", "close", "high", "low"]]

        else:
            tikData = pd.read_csv(data_file, parse_dates=[0])
            tikData.columns = [["time", "open", "close", "high", "low"]]

        #print(got_live)
        #print(f'Live columns: {tikData.columns}')
        #print(f'Downloaded columns: {tikData1.columns}')
        timeData2_ = tikData[['time']]
        timeData2 = timeData2_.to_numpy()
        last_timestamp = timeData2_.iloc[-1]
        data_ = tikData[['close']]
        data = data_.to_numpy()
        new_last_close = data_.iloc[-1]


        if frame == 'lowest_timeframe':
            if isinstance(last_timestamp[0], str):
                last_timestamp[0]=datetime.strptime(last_timestamp[0], '%Y-%m-%d %H:%M:%S')
            next_close_time = last_timestamp[0]+timedelta(hours=1)
        elif frame == 'middle_timeframe':
            if isinstance(last_timestamp[0], str):
                last_timestamp[0]=datetime.strptime(last_timestamp[0], '%Y-%m-%d %H:%M:%S')
            next_close_time = last_timestamp[0]+timedelta(hours=4)
        elif frame == 'highest_timeframe':
            if isinstance(last_timestamp[0], str):
                time_ = "00:00:00"
                last_timestamp[0]=datetime.strptime(f'{last_timestamp[0]} {time_}', '%Y-%m-%d %H:%M:%S')
            next_close_time = last_timestamp[0]+timedelta(days=1)

        # Getting training data size
        train_size = int(len(data) * 0.8)


        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)


        # Split into the training data and the actual data
        x_train = []
        y_train = []
        look_back = 60
        for i in trange(look_back, len(scaled_data), desc="Splitting training data"):
            x_train.append(scaled_data[i - 60:i, :])
            y_train.append(scaled_data[i, :])

        # Convert x_train and y_train to a numpy array
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Preparing variables for our LSTM
        #print(f'x train: {x_train}')
        #print(f'Train shape: {x_train.shape}')
        features = 1
        rows = x_train.shape[0]
        timesteps = x_train.shape[1]
        batch = 1  # 1 2 1459 2918

        # reshape data to a 3D array
        x_train = np.reshape(x_train, (rows, timesteps, features))

        pre_processed_data = {"x_train":x_train, "y_train":y_train, "batch":batch, 
                              "train_size":train_size, "timesteps":timesteps, "features":features, 
                              "scaler":scaler, "scaled_data": scaled_data, "look_back":look_back,
                              "last_trained_time":last_timestamp, 'last_close':new_last_close,
                              "next_close_time":next_close_time, "timeframe":timeframe, 
                              "c_pair":c_pair}
        return pre_processed_data

    def make_predictions(self, **kwargs):
        downloaded_data = kwargs['downloaded_data']
        model = kwargs['model']
        frame = kwargs['frame']

        if downloaded_data['train_or_predict'] == "predict" or downloaded_data['train_or_predict'] == "train":
            csv_file = downloaded_data["file"]
            look_back = downloaded_data['pre_processed_data']['look_back']

            # Get data
            if downloaded_data['got_live']:
                tikData = downloaded_data['data']
                tikData.columns = [["time", "open", "close", "high", "low"]]
            else:
                d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
                tikData = pd.read_csv(csv_file, parse_dates=[0], date_parser=d_parser, delimiter=',')
                tikData.columns = [["time", "open", "close", "high", "low"]]

            data = tikData[["close"]]
            data_ = tikData[["close"]]
            #data.drop_duplicates(subset ="date", keep=False, inplace=True)

            # Date preprocessing
            date_ = tikData[["time"]]  # Date type:  <class 'pandas._libs.tslibs.timestamps.Timestamp'>
            date = date_.to_numpy()
            # date = date2num(date)

            # Get the last data the length of the look back period
            last_look_back = data[-look_back:]
            date = date[-look_back:]
            data = last_look_back.to_numpy()

            # Scale data between 0 and 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Create an empty list
            X_data = []
            predicted_data = []
            predicted_data_ = []
            predicted_dates = []
            all_data = []
            all_dates = []

            # Append last look back data
            X_data.append(scaled_data)

            # Prepare a list that will contain all the data
            for y in range(len(scaled_data)):
                all_data.append(scaled_data[y][0])
                all_dates.append(date[y][0])#.to_pydatetime()

            # Convert to numpy
            X_data = np.array(X_data)
            dates = np.array(all_dates)
            original_data = X_data
            current_date = date_.iloc[-1]
            current_date = current_date[0]
            current_date_ = current_date
            current_price = data_.iloc[-1][0]


            # Loop through and keep getting new predicted data
            x = 0
            while x < 1:
                # Reshape to 3D Array
                X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

                # Get predicted scaled price
                predicted_price = model.predict(X_data)

                # Increment dates
                if frame == 'lowest_tiemframe':
                    current_date = current_date + timedelta(hours=1)
                elif frame == 'middle_timeframe':
                    if isinstance(current_date, str):
                        current_date = datetime.strptime(f'{current_date}', '%Y-%m-%d %H:%M:%S')
                    current_date = current_date + timedelta(hours=4)

                elif frame == 'highest_timeframe':
                    if isinstance(current_date, str):
                        current_date = datetime.strptime(f'{current_date} 00:00:00', '%Y-%m-%d %H:%M:%S')
                    current_date = current_date + timedelta(days=1)


                # Append predicted data where needed
                X_data = np.append(X_data, predicted_price[0][0])
                predicted_data.append(predicted_price[0][0])
                predicted_data_.append(predicted_price[0][0])
                all_data.append(predicted_price[0][0])
                predicted_dates.append(current_date)

                # Get the last look back and prepare data for next round
                X_data = X_data[-look_back:]

                X_data = np.reshape(X_data, (1, X_data.shape[0], 1))

                # Add one hour to the date array

                x = x + 1

            #print(f"\nData shape: {X_data.shape}")
            #print(f"Our predicted date: {predicted_dates[0]}")
            X_data = np.reshape(X_data, (X_data.shape[1]))

            '''
                predictions = scaler.inverse_transform(predictions)
                predict_df = pd.DataFrame(data=predictions, columns=["Predictions"])


                mse = np.sqrt(np.mean(predictions - y_test) ** 2)'''

            # Turn to numpy and shape data to be reshaped
            prediction = all_data[-121:]
            prediction = np.array(prediction)
            prediction = np.reshape(prediction, (prediction.shape[0], 1))
            num_all = np.array(all_data)
            num_all = np.reshape(num_all, (num_all.shape[0], 1))
            original_data = np.reshape(original_data, (original_data.shape[1], 1))
            predicted_dates = np.array(predicted_dates)
            predicted_dates_num = date2num(predicted_dates)
            predicted_data = np.array(predicted_data)
            predicted_data = np.reshape(predicted_data, (predicted_data.shape[0], 1))


            # Inverse transform
            real_data = scaler.inverse_transform(num_all)
            original_data = scaler.inverse_transform(original_data)
            all_data = scaler.inverse_transform(prediction)
            predicted_data = scaler.inverse_transform(predicted_data)



        else:
            print('\n No prediction made\n')

        return original_data, dates, all_data, predicted_dates, real_data, predicted_dates_num, predicted_data, current_price, current_date_

    def train_model(self, **kwargs):
        # Get relavent data
        pre_processed_data=kwargs['pre_processed_data']
        model = kwargs['model']

        x_train = pre_processed_data['x_train']
        y_train = pre_processed_data['y_train']
        batch = pre_processed_data['batch']
        timeframe=pre_processed_data['timeframe']
        c_pair = pre_processed_data['c_pair']

        # Saves the model every now and again incase something happens
        checkpoint = ModelCheckpoint('', save_best_only=True)

        # Getting our model ready to stop early if it stops improving
        ES = EarlyStopping()

        # Load existing model
        model = load_model(model)

        # Train the model
        model.fit(x_train, y_train,
                  batch_size=batch, epochs=1,
                  callbacks=[checkpoint, ES])  # shuffle=False
        # model.reset_states()

        # Saving the model for later use
        #print("Model created")
        model.save(c_pair + f"_lleno_close_{timeframe}.h5")

        return model



    def train_models(self, **kwargs):

        # figure out which model need to be trained
        downloaded_data = kwargs["downloaded_data"]
        frame = kwargs['frame']
        processed_data=kwargs['pre_processed_data']

        #print(f"""Train or Predict:{downloaded_data["train_or_predict"]}\nframe: {frame}""")
        if (downloaded_data["train_or_predict"] == "train" or downloaded_data["train_or_predict"] == "predict") and frame=="lowest_timeframe":
            timeframe = '1Hour'
            c_pair = "USDZAR"
            processed_data = self.preprocess_single_batch(data=downloaded_data['file'], pre_processed_data=processed_data, 
                                                          timeframe=timeframe, c_pair=c_pair, frame=frame, 
                                                          got_live=downloaded_data['got_live'], df_ready=downloaded_data['data'])
            new_model = self.train_model(model="USDZAR_lleno_close_1Hour.h5", pre_processed_data=processed_data)
                       
        elif (downloaded_data["train_or_predict"] == "train" or downloaded_data["train_or_predict"] == "predict") and frame=="middle_timeframe":
            timeframe = '4Hour'
            c_pair = "USDZAR"
            processed_data = self.preprocess_single_batch(data=downloaded_data['file'], pre_processed_data=processed_data, 
                                                          timeframe=timeframe, c_pair=c_pair, frame=frame, 
                                                          got_live=downloaded_data['got_live'], df_ready=downloaded_data['data'])
            new_model = self.train_model(model="USDZAR_lleno_close_4hour.h5", pre_processed_data=processed_data)
            

        elif (downloaded_data["train_or_predict"] == "train" or downloaded_data["train_or_predict"] == "predict") and frame=="highest_timeframe":
            timeframe = '1D'
            c_pair = "USDZAR"
            processed_data = self.preprocess_single_batch(data=downloaded_data['file'], pre_processed_data=processed_data, 
                                                          timeframe=timeframe, c_pair=c_pair, frame=frame, 
                                                          got_live=downloaded_data['got_live'], df_ready=downloaded_data['data'])
            new_model = self.train_model(model="USDZAR_lleno_close_1D.h5", pre_processed_data=processed_data)

        else:
            new_model = False
            processed_data = False

        # Record new last trained dates
        return new_model, processed_data
    
    def find_paths(self, **kwargs):
        documents = True
        metatrader = True

        rootdir = "C:\\Users\\"
        to_meta = "AppData\\Roaming\\MetaQuotes\\Terminal"
        more_meta = "MQL5\\Files"
        to_docs = "Documents\\Financial trading\\Datasets for machine learning\\Predictions made"
        paths = []

        for file in os.listdir(rootdir):
            d = f"{rootdir}{file}"
            if os.path.isdir(d):
                if os.path.exists(f"{d}\\{to_docs}"):
                    if documents:
                        paths.append(f"{d}\\{to_docs}")
                if os.path.exists(f"{d}\\{to_meta}"):
                    for f in os.listdir(f"{d}\\{to_meta}"):
                        dir = f"{d}\\{to_meta}\\{f}"
                        if os.path.isdir(dir):
                            if os.path.exists(f"{dir}\\{more_meta}"):
                                if metatrader:
                                    paths.append(f"{dir}\\{more_meta}")
  
        return paths
  
  
    def save_history(self, **kwargs):
        prediction = kwargs["prediction"]
        timeframe = kwargs["timeframe"]
        current_date_ = kwargs["current_date_"]
        filename = f"Historical data {timeframe}"
        paths = self.find_paths()
        
        # Choose where is save file according to the OS
        for path in paths:
            with open(f'{path}\\{filename}.txt', 'a') as f:
                for i in range(len(prediction)):
                    f.write(f"{current_date_},{str(prediction[i][0])}\n")

             
    def send_coordinates(self, **kwargs):
        prediction = kwargs["prediction"]
        timeframe = kwargs["timeframe"]
        paths = self.find_paths()

        # Choose where is save file according to the OS
        for path in paths:
            with open(f'{path}\\{timeframe}.txt', 'w') as f:
                for i in range(len(prediction)):
                    f.write(str(prediction[i][0]))

            with open(f'{path}\\{timeframe}.csv', mode='w') as file:
                file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                for i in range(len(prediction)):
                    #file_writer.writerow([current_date_, current_price, predicted_dates[i], prediction[i][0]])
                    file_writer.writerow([prediction[i][0]])

                print('\nNew prediction saved and sent to Metatrader!!\n')

        
    
    def update_variables(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        downloaded_data = kwargs['downloaded_data']
        frame = kwargs['frame']
        pre_processed_data = kwargs["pre_processed_data"]
        data_timedelta = kwargs['data_timedelta']

        """"x_train":x_train, "y_train":y_train, "batch":batch,
        "train_size":train_size, "timesteps":timesteps, "features":features, 
        "scaler":scaler, "scaled_data": scaled_data, "look_back":look_back,
        "last_trained_time":last_timestamp, 'last_close':new_last_close,
        "next_close_time":next_close_time"""


        pre_processed_data[frame]['x_train'] = downloaded_data["pre_processed_data"]["x_train"]
        pre_processed_data[frame]['y_train'] = downloaded_data["pre_processed_data"]["y_train"]
        pre_processed_data[frame]['batch'] = downloaded_data["pre_processed_data"]["batch"]

        pre_processed_data[frame]['train_size'] = downloaded_data["pre_processed_data"]["train_size"]
        pre_processed_data[frame]['timesteps'] = downloaded_data["pre_processed_data"]["timesteps"]
        pre_processed_data[frame]['features'] = downloaded_data["pre_processed_data"]["features"]

        pre_processed_data[frame]['scaler'] = downloaded_data["pre_processed_data"]["scaler"]
        pre_processed_data[frame]['scaled_data'] = downloaded_data["pre_processed_data"]["scaled_data"]
        pre_processed_data[frame]['look_back'] = downloaded_data["pre_processed_data"]["look_back"]

        pre_processed_data[frame]['last_trained_time'] = downloaded_data["pre_processed_data"]["last_trained_time"]
        pre_processed_data[frame]['last_close'] = downloaded_data["pre_processed_data"]["last_close"]
        pre_processed_data[frame]['next_close_time'] = downloaded_data["pre_processed_data"]["next_close_time"]


        pre_processed_data[frame]['c_pair'] = downloaded_data["pre_processed_data"]["c_pair"]
        pre_processed_data[frame]['timeframe'] = downloaded_data["pre_processed_data"]["timeframe"]

        return pre_processed_data

    def save_new_pre_processed_data(self, **kwargs):
        pre_processed_data = kwargs['pre_processed_data']

        with open('trained_history.pickle', "wb") as f:
            pickle.dump(pre_processed_data, f)
            print('\nNew findings saved!!\n')


    def rename_downloaded_files(self, **kwargs):
        
        for file in os.listdir('Currency pair//'):
            if file[-1] == "v":
                if os.path.exists('Currency pair//'+file):
                    os.remove('Currency pair//'+file)
                else:
                    print("The file does not exist")
                

    def start_trade_loop(self, **kwargs):
        #['x_train', 'y_train', 'batch', 'train_size', 'timesteps', 'features', 'scaler', 'scaled_data', 'c_pair', 'mse', 'predict_df', 'timeframe', 'last_trained_time', 'last_close', 'next_close_date']
        # Required variables
        data_timedelta = kwargs["data_timedelta"]
        pre_processed_data=kwargs["pre_processed_data"]
        loop_count = 0


        # THE LOOP BEGINS
        while True:
            # Just delete all files before starting
            self.rename_downloaded_files()

            # Download data required
            data_downloaded = False
            downloaded_data = {}
            for k, v in pre_processed_data.items():
                downloaded_data_, dd = self.download_data(data_timedelta=data_timedelta, 
                                pre_processed_data=pre_processed_data,
                                loop_count=loop_count, frame=k)
                
                downloaded_data[k]=downloaded_data_[k]

                #print(f'dd: {dd}\n key: {k}')
                if dd == True:
                    data_downloaded = True

            if data_downloaded:
                for key, val in downloaded_data.items():
                    if downloaded_data[key]["downloaded"] == True and downloaded_data[key]["got_live"]:
                        # Decide whether a prediction or a traning is required. Perform each task as per usual
                        new_model, processed_data = self.train_models(downloaded_data=downloaded_data[key], frame=key, pre_processed_data=pre_processed_data)

                        if new_model:
                            downloaded_data[key]['pre_processed_data'] = processed_data
                            original_data, dates, all_data, predicted_dates, real_data, predicted_dates_num, prediction, current_price, current_date_ = self.make_predictions(downloaded_data=downloaded_data[key], 
                                                                                                                                    model=new_model,
                                                                                                                                    frame=key)
                            """
                            if key == "middle_timeframe":    
                                now = datetime.now()
                                current_date_ = f'{now.year}-{now.month}-{now.day} {now.hour}:00:00'
                                current_date_ = datetime.strptime(current_date_, '%Y-%m-%d %H:%M:%S')"""

                            # Send to Metatrader
                            self.send_coordinates(prediction=prediction, timeframe=key)

                            # Save to historical file
                            self.save_history(prediction=prediction, current_date_=current_date_,
                                                timeframe=key)

                            # Update variables
                            pre_processed_data = self.update_variables(data_timedelta=data_timedelta,
                                                pre_processed_data=pre_processed_data, downloaded_data=downloaded_data[key],
                                                frame=key)

                        else:
                            print(f'No Model for frame: {key}')

                # Files need to be purged renamed or relaocated
                self.rename_downloaded_files()

                # Save new developments
                self.save_new_pre_processed_data(pre_processed_data=pre_processed_data)
                
            else:
                print("\nNo training or prediction required\n")
            
            # wait to make the next prediction
            print("\n5 minute wait ...\n")
            sleep(300)

            # Variables for next iteration
            data_timedelta = self.get_dates(pre_processed_data=pre_processed_data)
            loop_count = loop_count+1

