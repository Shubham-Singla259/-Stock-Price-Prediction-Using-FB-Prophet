from taipy.gui import Gui
import yfinance as yf
from prophet import Prophet
from datetime import date
import pandas as pd


start_date = "2010-01-01"
end_date = date.today().strftime("%Y-%m-%d")
selected_stock = "RELIANCE.NS"
n_years = 1

def get_stock_data(ticker, start, end):
    ticker_data = yf.download(ticker, start, end)
    ticker_data.reset_index(inplace=True) # Put date in first column
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date']).dt.tz_localize(None)
    return ticker_data


def generate_forecast_data(data, n_years):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

    m=Prophet(daily_seasonality=True)
    m.fit(df_train)
    future=m.make_future_dataframe(periods=365)
    fc=m.predict(future)[['ds','yhat_lower','yhat_upper']].rename(columns={"ds": "Date", "yhat_lower":"Lower", "yhat_upper":"Upper"})
    return fc

# Get Data and forecast
data = get_stock_data(selected_stock, start_date, end_date)
forecast = generate_forecast_data(data, n_years)

page = '''
# Stock Price Forecasting Project

<|{forecast}|chart|mode=line|x=Date|y[1]=Lower|y[2]=Upper|>
'''

Gui(page).run(run_browser=False)
