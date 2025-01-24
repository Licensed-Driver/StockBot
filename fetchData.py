import yfinance as yf
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import csv as csv
import threading
import numpy as np

def saveFortune500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    # Source: https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy
    # Note: One of the included holdings is CASH_USD.
    url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
    tickers = pd.read_excel(url, engine='openpyxl', index_col='Ticker', skiprows=4).dropna()
    indices = tickers.index
    tickerList = indices.tolist()
    tickerList.remove("-")
    tickerList.remove("BRK.B")
    tickerList.remove("MRP-W")
    tickerList.remove("BF.B")
    with open("tickerList.csv", "w") as file:
        tickerList = tickerList.__str__()
        tickerList = tickerList.replace(' ', '')
        tickerList = tickerList.replace('[', '')
        tickerList = tickerList.replace(']', '')
        tickerList = tickerList.replace("'", '')
        file.write(tickerList)
        file.close()
    return



def getTickerInfo():
    with open("tickerList.csv", "r") as file:
        lines = pd.read_csv(file)
        tickers = lines.columns.tolist()
        file.close()
    now = dt.now()
    max = now - td(days = 8)
    tickerInfo = yf.download(tickers,start=max.strftime("%Y"+"-"+"%m"+"-"+"%d"), interval="1m", end=now.strftime("%Y"+"-"+"%m"+"-"+"%d"))
    return tickerInfo

def saveTickerInfo():
    tickDat = getTickerInfo()
    high = tickDat.High
    low = tickDat.Low
    pd.options.mode.use_inf_as_na = True
    avg = (high + low)/2
    avg.interpolate(inplace=True)
    avg.bfill(inplace=True)
    avg.ffill(inplace=True)
    avg.to_csv("Fortune500_PPM_{}.csv".format(dt.now().strftime("%Y"+"-"+"%m"+"-"+"%d")), date_format="%Y-%m-%d-%H-%M")
    return avg

def normalizePrices(dateTime=dt.now()):
    prices = pd.read_csv("Fortune500_PPM_{}.csv".format(dateTime.strftime("%Y"+"-"+"%m"+"-"+"%d")), index_col="Datetime", date_format="%Y-%m-%d-%H-%M", dtype="float64")
    prices.interpolate(inplace=True)
    threads = []
    counter = 0
    for col in prices.columns:
        threads.append(threading.Thread(target=normalizeColumn, args=(prices[col],)).start())
        print(counter)
        counter+=1
    counter = 1
    for col in threads:
        print(counter)
        count+=1
        col.join()
    print(prices)
        

def normalizeColumn(column):
    prevPrice = column.iloc[0]
    column.iloc[0] = 0
    for num in range(1, column.size):
        perChange = ((column.iloc[num] - prevPrice)/prevPrice) * 100
        prevPrice = column.iloc[num]
        column.iloc[num] = perChange
        
saveTickerInfo()
normalizePrices()