import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def reshaper(arr):
    arr = np.array(np.split(arr, 3, axis=1))
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr

def scalar_normalize(train_data, test_data):
    scaler = RobustScaler()
    scaler.fit(train_data[:, 2:-2])
    train_data[:, 2:-2] = scaler.transform(train_data[:, 2:-2])
    test_data[:, 2:-2] = scaler.transform(test_data[:, 2:-2])

# perc=[0.5, 0.5] chia tỉ lệ 50% 50% (một nửa rank cao nhất sẽ label 1) (nửa còn lại label 0)
def create_label_LSTM_Intraday(df_open, df_close, perc=[0.5, 0.5]):
    if not np.all(df_close['Date'] == df_open['Date']):
        raise ValueError('Date index mismatch')

    # list(np.cumsum(perc)) => perc=[0.5, 0.5] -> [0.5, 1.0]
    # 0.0 -> 0.5 sẽ label 0
    # 0.5 -> 1.0 sẽ label 1
    perc = [0.] + list(np.cumsum(perc))
    label = (df_close.iloc[:, 1:] / df_open.iloc[:, 1:] - 1).apply(
        lambda x: pd.qcut(x.rank(method='first'), perc, labels=False), axis=1)

    return label[1:]  # bỏ ngày đầu tiên vì không có giá trị trước đó

def create_label_LSTM_NextDay(df_close, perc=[0.5, 0.5]):
    perc = [0.] + list(np.cumsum(perc))
    label = df_close.iloc[:, 1:].pct_change(fill_method=None)[1:].apply(
        lambda x: pd.qcut(x.rank(method='first'), perc, labels=False), axis=1)

    return label

def create_stock_data_LSTM_Intraday_3f(df_open, df_close, label, ticker: str, test_year: int, window: int = 240):
    df = pd.DataFrame()
    df['Date'] = df_close['Date']
    df['Name'] = ticker

    daily_change = df_close[ticker] / df_open[ticker] - 1
    nextday_ret = df_open[ticker].shift(-1) / df_close[ticker] - 1
    close_change = df_close[ticker].pct_change()

    # Tạo dict cho từng nhóm đặc trưng
    intra_features = {f'IntraR{k}': daily_change.shift(k) for k in range(window)[::-1]}
    next_features = {f'NextR{k}': nextday_ret.shift(k) for k in range(window)[::-1]}
    close_features = {f'CloseR{k}': close_change.shift(k) for k in range(window)[::-1]}

    # Gộp tất cả vào DataFrame
    df = pd.concat([df,
                    pd.DataFrame(intra_features),
                    pd.DataFrame(next_features),
                    pd.DataFrame(close_features)],
                   axis=1)

    df['IntraR-future'] = daily_change.shift(-1)
    df['label'] = label[ticker].values.tolist() + [np.nan]
    df['Month'] = df['Date'].str[:7]

    df = df.dropna()
    df['Year'] = df['Month'].str[:4].astype(int)

    train = df[df['Year'] < test_year].drop(columns=['Month', 'Year'])
    test = df[df['Year'] == test_year].drop(columns=['Month', 'Year'])
    return np.array(train), np.array(test)

def create_stock_data_LSTM_Intraday_1f(df_open, df_close, label, ticker: str, test_year: int, window: int = 240):
    df = pd.DataFrame()
    df['Date'] = df_close['Date']
    df['Name'] = ticker

    daily_change = df_close[ticker] / df_open[ticker] - 1

    # Tạo dict cho từng nhóm đặc trưng
    intra_features = {f'IntraR{k}': daily_change.shift(k) for k in range(window)[::-1]}

    # Gộp tất cả vào DataFrame
    df = pd.concat([df,
                    pd.DataFrame(intra_features)],
                   axis=1)

    df['IntraR-future'] = daily_change.shift(-1)
    df['label'] = label[ticker].values.tolist() + [np.nan]
    df['Month'] = df['Date'].str[:7]

    df = df.dropna()
    df['Year'] = df['Month'].str[:4].astype(int)

    train = df[df['Year'] < test_year].drop(columns=['Month', 'Year'])
    test = df[df['Year'] == test_year].drop(columns=['Month', 'Year'])
    return np.array(train), np.array(test)

def create_stock_data_LSTM_NextDay_1f(df_close, label, ticker: str, test_year: int, window: int = 240):
    df = pd.DataFrame()
    df['Date'] = df_close['Date']
    df['Name'] = ticker

    daily_change = df_close[ticker].pct_change()

    # Tạo dict cho từng nhóm đặc trưng
    intra_features = {f'R{k}': daily_change.shift(k) for k in range(window)[::-1]}

    # Gộp tất cả vào DataFrame
    df = pd.concat([df,
                    pd.DataFrame(intra_features)],
                   axis=1)

    df['R-future'] = daily_change.shift(-1)
    df['label'] = label[ticker].values.tolist() + [np.nan]
    df['Month'] = df['Date'].str[:7]

    df = df.dropna()
    df['Year'] = df['Month'].str[:4].astype(int)

    train = df[df['Year'] < test_year].drop(columns=['Month', 'Year'])
    test = df[df['Year'] == test_year].drop(columns=['Month', 'Year'])
    return np.array(train), np.array(test)

def create_label_RF_Intraday(df_open, df_close, perc=[0.5, 0.5]):
    if not np.all(df_close['Date'] == df_open['Date']):
        raise ValueError('Date index mismatch')

    perc = [0.] + list(np.cumsum(perc))
    label = (df_close.iloc[:, 1:] / df_open.iloc[:, 1:] - 1).apply(
        lambda x: pd.qcut(x.rank(method='first'), perc, labels=False), axis=1)

    return label

def create_stock_data_RF_Intraday_3f(df_open, df_close, label, ticker: str, test_year: int, window: int = 240):
    df = pd.DataFrame([])
    df['Date'] = list(df_close['Date'])
    df['Name'] = ticker

    daily_change = df_close[ticker] / df_open[ticker] - 1
    m = list(range(1, 20)) + list(range(20, 241, 20))

    for k in m:
        df['IntraR'+str(k)] = daily_change.shift(k)
    for k in m:
        df['CloseR'+str(k)] = df_close[ticker].pct_change(k).shift(1)
    for k in m:
        df['OverNR'+str(k)] = df_open[ticker]/df_close[ticker].shift(k)-1

    df['R-future'] = daily_change
    df['label'] = list(label[ticker])
    df['Month'] = list(df_close['Date'].str[:-3])

    df = df.dropna()
    trade_year = df['Month'].str[:4]
    df = df.drop(columns=['Month'])
    train_data = df[trade_year < str(test_year)]
    test_data = df[trade_year == str(test_year)]
    return np.array(train_data), np.array(test_data)


def create_stock_data_RF_Intraday_1f(df_open, df_close, label, ticker: str, test_year: int, window: int = 240):
    df = pd.DataFrame([])
    df['Date'] = list(df_close['Date'])
    df['Name'] = ticker

    daily_change = df_close[ticker] / df_open[ticker] - 1
    m = list(range(1, 20)) + list(range(20, 241, 20))

    for k in m:
        df['IntraR' + str(k)] = df_close[ticker].shift(1) / df_open[ticker].shift(k) - 1

    df['R-future'] = daily_change
    df['label'] = list(label[ticker])
    df['Month'] = list(df_close['Date'].str[:-3])
    df = df.dropna()

    trade_year = df['Month'].str[:4]
    df = df.drop(columns=['Month'])
    train_data = df[trade_year < str(test_year)]
    test_data = df[trade_year == str(test_year)]
    return np.array(train_data), np.array(test_data)