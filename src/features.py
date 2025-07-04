import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# perc=[0.5, 0.5] chia tỉ lệ 50% 50% (một nửa rank cao nhất sẽ label 1) (nửa còn lại label 0)
def create_label(df_open, df_close, perc=[0.5, 0.5]):

    # if not np.all(df_close.iloc[:, 0] == df_open.iloc[:, 0]):
    if not np.all(df_close['Date'] == df_open['Date']):
        raise ValueError('Date index mismatch')

    # list(np.cumsum(perc)) => perc=[0.5, 0.5] -> [0.5, 1.0]
    # 0.0 -> 0.5 sẽ label 0
    # 0.5 -> 1.0 sẽ label 1
    perc = [0.] + list(np.cumsum(perc))
    label = (df_close.iloc[:, 1:] / df_open.iloc[:, 1:] - 1).apply(
        lambda x: pd.qcut(x.rank(method='first'), perc, labels=False), axis=1)

    return label[1:]  # bỏ ngày đầu tiên vì không có giá trị trước đó

def create_stock_data(df_open, df_close, label, ticker: str, test_year: int, window: int = 240):
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


def scalar_normalize(train_data, test_data):
    scaler = RobustScaler()
    scaler.fit(train_data[:, 2:-2])
    train_data[:, 2:-2] = scaler.transform(train_data[:, 2:-2])
    test_data[:, 2:-2] = scaler.transform(test_data[:, 2:-2])