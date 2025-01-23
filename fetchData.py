import yfinance as yf
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import csv as csv

def saveSP500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    # Source: https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy
    # Note: One of the included holdings is CASH_USD.
    url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'
    tickers = pd.read_excel(url, engine='openpyxl', index_col='Ticker', skiprows=4).dropna()
    indices = tickers.index
    tickerList = indices.tolist()
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
        for n in range(1, 5):
            tickers.pop()
        tickers.remove("BRK.B")
        tickers.remove("-")
        file.close()
    now = dt.now()
    max = now - td(days = 1)
    tickerInfo = yf.download(tickers,start=max.strftime("%Y"+"-"+"%m"+"-"+"%d"), interval="1m", end=now.strftime("%Y"+"-"+"%m"+"-"+"%d"))
    return tickerInfo

def saveTickerInfo():
    tickDat = getTickerInfo()
    high = tickDat.High
    low = tickDat.Low
    avg = (high + low)/2
    avg.to_csv("SP500_PPM_{}.csv".format(dt.now().strftime("%Y"+"-"+"%m"+"-"+"%d")), date_format="%Y-%m-%d-%H-%M")
    return avg

saveTickerInfo()