import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def compute_technical_indicators(df_close, period_sma=20, period_rsi=14, period_bb=20):
    indicators = pd.DataFrame()
    indicators['Date'] = df_close['Date']

    for ticker in df_close.columns[1:]:  # bỏ cột 'Date'
        price = df_close[ticker]

        # SMA
        sma = price.rolling(window=period_sma).mean()
        indicators[f'{ticker}_SMA{period_sma}'] = sma

        # RSI
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period_rsi).mean()
        avg_loss = loss.rolling(window=period_rsi).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        indicators[f'{ticker}_RSI{period_rsi}'] = rsi

        # Bollinger Bands
        sma_bb = price.rolling(window=period_bb).mean()
        std_bb = price.rolling(window=period_bb).std()
        upper_bb = sma_bb + 2 * std_bb
        lower_bb = sma_bb - 2 * std_bb
        indicators[f'{ticker}_BB_upper'] = upper_bb
        indicators[f'{ticker}_BB_lower'] = lower_bb

    return indicators


def fetch_yfinance_data(tickers, start_date, end_date, include_technical_indicators=False):
    # Tính ngày bắt đầu có thêm buffer
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30)
    start_buffer = start_dt.strftime("%Y-%m-%d")

    data = yf.download(
        tickers=' '.join(tickers),
        start=start_buffer,
        end=end_date,
        progress=False,
        auto_adjust=True,
        group_by='ticker',
        threads=True
    )

    open_list, close_list = [], []
    valid_tickers = []

    for ticker in tickers:
        ticker_data = data.get(ticker)
        if ticker_data is None:
            continue
        if ticker_data['Open'].dropna().empty or ticker_data['Close'].dropna().empty:
            continue

        valid_tickers.append(ticker)
        open_list.append(data[ticker]['Open'].rename(ticker))
        close_list.append(data[ticker]['Close'].rename(ticker))

    print(f"[INFO] fetch_yfinance_data(): Lấy thành công dữ liệu cho {len(valid_tickers)}/{len(tickers)} mã.")

    if not valid_tickers:
        return pd.DataFrame(), pd.DataFrame()

    # Dùng concat một lần thay vì insert tuần tự (tránh fragmented warning)
    df_open = pd.concat(open_list, axis=1)
    df_close = pd.concat(close_list, axis=1)

    # Chuyển index thành Date
    df_open.insert(0, 'Date', df_open.index.strftime('%Y-%m-%d'))
    df_close.insert(0, 'Date', df_close.index.strftime('%Y-%m-%d'))

    # Cắt bỏ buffer
    mask = df_close['Date'] >= start_date
    df_open = df_open[mask].reset_index(drop=True)
    df_close = df_close[mask].reset_index(drop=True)

    if include_technical_indicators:
        df_tech = compute_technical_indicators(df_close)
        df_tech = df_tech[df_tech['Date'] >= start_date].reset_index(drop=True)
        return df_open, df_close, df_tech
    else:
        return df_open, df_close

