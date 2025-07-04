import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_yfinance_data(tickers, start_date, end_date):
    df_open, df_close = pd.DataFrame(), pd.DataFrame()
    for ticker in tickers:
        # Lấy dữ liệu từ yfinance (chỉ lấy Open và Close)
        # Mỗi df_open của ticker chứa giá Open
        # Tương tự với df_close
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            continue
        df_open[ticker] = data['Open']
        df_close[ticker] = data['Close']

    # Chuyển từng cổ phiếu thành cột
    df_open.insert(0, 'Date', df_close.index.strftime('%Y-%m-%d'))
    df_close.insert(0, 'Date', df_close.index.strftime('%Y-%m-%d'))

    return df_open.reset_index(drop=True), df_close.reset_index(drop=True)


