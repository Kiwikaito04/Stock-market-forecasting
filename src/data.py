import yfinance as yf
import pandas as pd

def fetch_yfinance_data(tickers, start_date, end_date):
    # Dùng join(',') để gộp thành chuỗi cho yfinance tải batch
    data = yf.download(
        tickers=' '.join(tickers),
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        group_by='ticker',
        threads=True
    )

    df_open, df_close = pd.DataFrame(), pd.DataFrame()

    for ticker in tickers:
        if ticker not in data.columns.levels[0]:
            continue  # ticker không có dữ liệu
        df_open[ticker] = data[ticker]['Open']
        df_close[ticker] = data[ticker]['Close']

    df_open.insert(0, 'Date', df_close.index.strftime('%Y-%m-%d'))
    df_close.insert(0, 'Date', df_close.index.strftime('%Y-%m-%d'))

    return df_open.reset_index(drop=True), df_close.reset_index(drop=True)
