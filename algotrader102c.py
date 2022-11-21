# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 20:39:55 2021

@author: LK
"""
import numpy as np
import math
import MetaTrader5 as mt5
from datetime import datetime
from time import sleep
# from time import time
import time
# import schedule
import datetime
import pandas as pd
from talib import *
import pytz
from pylab import mpl, plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# counter = 0
down_count = 0
time_wait = 1800  #time to wait to loop the code
def task(curr_pair, def_pips):
# account = 51974070  zucbxb5x 50676291 UVPZ5hKr    FxPro-MT5
    account = 5400730
    
    mt5.initialize()
    # authorized=mt5.login(account, password="UVPZ5hKr", server = "ICMarketsSC-Demo")
    authorized=mt5.login(account, password="MY6POwYY", server = "FxPro-MT5")
    
    if authorized:
        print("Connected: Connecting to MT5 Client")
    else:
        print("Failed to connect at account #{}, error code: {}"
              .format(account, mt5.last_error()))
    
    timezone = pytz.timezone("Etc/UTC")
    
    hours_added = datetime.timedelta(hours = 2)
    
    
    
    now = datetime.datetime.now()
    now = now + hours_added
    
    yr =int(now.strftime("%Y"))
    mn = int(now.strftime("%m"))
    dy = int(now.strftime("%d"))
    hr = int(now.strftime("%H"))
    minu = int(now.strftime("%M"))
    sec = int(now.strftime("%S"))
    
    past_ticks = 2000 # number of historical ticks to use to test
    time_frame = mt5.TIMEFRAME_M30
    # curr_pair = "GBPUSD" #EURUSD BTCUSD AUDUSD USDJPY GBPUSD
    
    utc_from = datetime.datetime(yr, mn, dy, hr, minu, sec, tzinfo=timezone)
    # utc_to = datetime(2021, 1, 10)
    if curr_pair == "BTCUSD":
        past_ticks = 1000
        time_frame = mt5.TIMEFRAME_M15
    rates = mt5.copy_rates_from(curr_pair, time_frame, utc_from, past_ticks)
    # rates = mt5.copy_ticks_from("EURUSD", utc_from, 100000, mt5.COPY_TICKS_ALL)
        
    rates_frame = pd.DataFrame(rates)
    
    # convert time in seconds into the datetime format
    try:
        rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    except KeyError:
        # down_count = down_count+1
        print("ERROR: Check Internet Connection!!! Missed: ")
        sleep(500)
        starter()
    rf = rates_frame
    rf = rf.drop(['spread'], axis = 1)
    rf = rf.drop(['real_volume'], axis = 1)
    rf = rf.drop(['tick_volume'], axis = 1)
    rf['cci'] = CCI(rf['high'], rf['low'], rf['close'],timeperiod = 14)
    rf['up'], rf['mid'], rf['low'] = BBANDS(rf['close'], timeperiod=20)
    rf['fast'], rf['slow'], rf['signal'] = MACD(rf['close'])
    rf['rsi'] = RSI(rf['close'], timeperiod=14)
    
    rf['OH'] = rf['open'] - rf['high']
    rf['OL'] = rf['open'] - rf['low']
    
    
    # determines the numer of decimal places for the pips 
    # def_pips = -3
    
    # Get OH and OL columns in place based on the conditions of 200 pips gain
    cond_1 = [
        (rf['OH'] < -1.2*10**def_pips),
        (rf['OH'] >= -1.2*10**def_pips)
        ]
    
    cond_2 = [
        (rf['OL'] > 2*10**def_pips),
        (rf['OL'] <= 2*10**def_pips)
        ]
    
    val_1 = [1,0]
    val_2 = [2,0]
    
    rf['R1'] = np.select(cond_1, val_1)
    rf['R2'] = np.select(cond_2, val_2)
    
    # delete first set of 15 rows of data cos of CCI 14 period
    rf = rf.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    # Lag the data behind for prediction
    rf['R1'] = rf['R1'].shift(-1)
    rf['R2'] = rf['R2'].shift(-1)
    
    # Model
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=1 * [50], random_state=1)
    model = XGBClassifier()
    
    data2 = rf.drop(['OH','OL'], axis=1)
    data2 = data2.set_index('time')
    x = data2.drop(['R1','R2'], axis = 1)
    row_pred = x.iloc[(-1):]
    x = x[:-1]
    data2 = data2[:-1]
    y = pd.DataFrame(data2['R1']) # R1 indicate for OH - BUY
    y = y.astype(np.float64)
    y_Low = pd.DataFrame(data2['R2']) #R2 indicates OL - SELL
    y_Low = y_Low.astype(np.float64)
    
    
    # last_row = len(data2)
    
    # row_pred = x.iloc[(-1):]
    # row_pred = row_pred.set_index('time')
    
    
    
    # Split into training and testing data
    # split to train and test for BUY
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, shuffle=False)
    # Split to train test for SELL
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x,y_Low, test_size= 0.25, shuffle=False)
    
    
    # y_train = pd.DataFrame(y_train)
    # y_test = pd.DataFrame(y_test)
    
    # model training and testing for buy accuracy
    y_train = y_train.astype(np.float64)
    x_train = x_train.astype(np.float64)
    model.fit(x_train, y_train)
    yB = model.predict(x_test)
    z1 =model.feature_importances_
    y_Buy = model.predict(row_pred)
    # y_Buy = model.predict(x_live1)
    
    # model training and testing for sell accuracy
    model.fit(x_train1, y_train1)
    yS = model.predict(x_test1)
    z2 =model.feature_importances_
    y_Sell = model.predict(row_pred)
    # y_Sell = model.predict(x_live1)
    
    Buy_accuracy = accuracy_score(y_test, yB)
    Sell_accuracy =accuracy_score(y_test1, yS)
    
    plt.barh(x_test.columns, model.feature_importances_)
    plt.barh(x_test1.columns, model.feature_importances_)
    
    print ("\a") # alarm
    
    a = (row_pred,' Buy - ',y_Buy[-1] , 'Sell - ',y_Sell[-1])
    print(row_pred,' Buy ',y_Buy[-1] ,' acc ',Buy_accuracy, ' Sell ',y_Sell[-1], ' acc ', Sell_accuracy)
    print ("\a")
    
    # prepare the buy request structure
    symbol = curr_pair
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found, can not call order_check()")
        mt5.shutdown()
        quit()
     
    # if the symbol is unavailable in MarketWatch, add it
    if not symbol_info.visible:
        print(symbol, "is not visible, trying to switch on")
        if not mt5.symbol_select(symbol,True):
            print("symbol_select({}}) failed, exit",symbol)
            mt5.shutdown()
            quit()
     
    lot = 0.01
    tot_orders=mt5.positions_total()
    point = mt5.symbol_info(symbol).point
    if curr_pair == "BTCUSD":
        point = mt5.symbol_info(curr_pair).point*100
    
    price = mt5.symbol_info_tick(symbol).ask
    price2 = mt5.symbol_info_tick(symbol).bid
    deviation = 2
    B_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - (150 * point),
        "tp": price + (120 * point),
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
     
    S_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price2,
        "sl": price + (250 * point),
        "tp": price - (180 * point),
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # check for time difference between last position and next position
    
    
    df_pos = mt5.positions_get()
    
    # check if currency already exist in open trades
    df_posa = pd.DataFrame(list(df_pos),columns=df_pos[0]._asdict().keys())
    if curr_pair in df_posa.values:
        curr_exist = 1
    else:
        curr_exist = 0
        
    if len(df_pos) > 0:
        df_pos = pd.DataFrame(list(df_pos),columns=df_pos[0]._asdict().keys())
        # df_pos['time'] = pd.to_datetime(df_pos['time'], unit='s')
        df_pos_max = df_pos[df_pos.ticket == df_pos.ticket.max()]
        time_dif = abs(int((round(time.time())) - df_pos_max.time))    
    else:
        time_dif = time_wait
    
    # check to send a request
    if tot_orders<6 and time_dif>time_wait/2 and curr_exist == 0:
    
    # send a buy request
        if y_Buy[-1]==1.0 and y_Sell[-1]==0.0 and Buy_accuracy>0.7:
        
            result = mt5.order_send(B_request)
            print("BUY TRADE DONE")
        
        
        # send a sell request
        if y_Buy[-1]==0.0 and y_Sell[-1]==2.0 and Sell_accuracy>0.7:
            result = mt5.order_send(S_request)
            print("SELL TRADE DONE")
def starter():        
    while True:

    	# thing to run
        for curr_picker in [1,2,3,4]:
        # if curr_picker == 1:
            if curr_picker == 1:
                curr_pair = "GBPUSD" #EURUSD BTCUSD AUDUSD USDJPY GBPUSD
                def_pips = -3
                # symbol_info = mt5.symbol_info("GBPUSD")
                # point = mt5.symbol_info("GBPUSD").point
                
                task(curr_pair, def_pips)
                # curr_picker = curr_picker+1
                # break;
            elif curr_picker == 2:
                curr_pair = "EURUSD" #EURUSD BTCUSD AUDUSD USDJPY GBPUSD
                def_pips = -3
                # symbol_info = mt5.symbol_info(curr_pair)
                # point = mt5.symbol_info(curr_pair).point
                task(curr_pair, def_pips)
                # curr_picker = curr_picker+1
                # break;
            elif curr_picker == 3:
                curr_pair = "AUDUSD" #EURUSD BTCUSD AUDUSD USDJPY GBPUSD
                def_pips = -3
                # symbol_info = mt5.symbol_info(curr_pair)
                # point = mt5.symbol_info(curr_pair).point
                task(curr_pair, def_pips)
                # curr_picker = curr_picker+1
                # break;
        
            elif curr_picker == 4:
                curr_pair = "BTCUSD" #EURUSD BTCUSD AUDUSD USDJPY GBPUSD
                def_pips = 2
                # symbol_info = mt5.symbol_info(curr_pair)
                # point = mt5.symbol_info(curr_pair).point*1000
                task(curr_pair, def_pips)
                # curr_picker = 1
                # break;
        # task()
        aa = time_wait - (math.floor(time.time() % time_wait))
        bb = pd.to_datetime(math.ceil(time.time() + aa), unit='s')
        print("Next runtime: ", bb)
        sleep(aa)
            
            
starter()

