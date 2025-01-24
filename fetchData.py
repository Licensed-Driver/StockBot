import yfinance as yf
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import csv as csv
import threading
import numpy as np
import yfinance.shared as shared

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
    max = now - td(days = 1)
    while(True):
        tickerInfo = yf.download(tickers,start=max.strftime("%Y"+"-"+"%m"+"-"+"%d"), interval="1m", end=now.strftime("%Y"+"-"+"%m"+"-"+"%d"))
        if(shared._ERRORS.keys().__len__() == 0):
            break
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
    now = dt.now()
    avg.to_csv("PPM/{}/{}/Fortune500_PPM_{}.csv".format(now.strftime("%Y"), now.strftime("%m"), now.strftime("%Y"+"-"+"%m"+"-"+"%d")), date_format="%Y-%m-%d-%H-%M")
    return avg

def normalizePrices(dateTime=dt.now()):
    now = dt.now()
    prices = pd.read_csv("PPM/{}/{}/Fortune500_PPM_{}.csv".format(now.strftime("%Y"), now.strftime("%m"), now.strftime("%Y"+"-"+"%m"+"-"+"%d")), index_col="Datetime", date_format="%Y-%m-%d-%H-%M", dtype="float64")
    prices.interpolate(inplace=True)
    threads = []
    for col in prices.columns:
        thread = threading.Thread(target=normalizeColumn, args=(prices[col],))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    prices.to_csv("PCPM/{}/{}/Fortune500_PCPM_{}.csv".format(now.strftime("%Y"), now.strftime("%m"), now.strftime("%Y"+"-"+"%m"+"-"+"%d")), date_format="%Y-%m-%d-%H-%M")
        

def normalizeColumn(column):
    prevPrice = column.iloc[0]
    column.iloc[0] = 0
    for num in range(1, column.size):
        perChange = ((column.iloc[num] - prevPrice)/prevPrice) * 100
        prevPrice = column.iloc[num]
        column.iloc[num] = perChange
        
saveTickerInfo()
normalizePrices()