


def create_price_features(df_open, df_close, ticker: str, window: int = 240):
    daily_change = df_close[ticker] / df_open[ticker] - 1
    nextday_ret = df_open[ticker].shift(-1) / df_close[ticker] - 1
    close_change = df_close[ticker].pct_change(fill_method=None)

    intra_features = {f'IntraR{k}': daily_change.shift(k) for k in range(window)[::-1]}
    next_features = {f'NextR{k}': nextday_ret.shift(k) for k in range(window)[::-1]}
    close_features = {f'CloseR{k}': close_change.shift(k) for k in range(window)[::-1]}

    return intra_features, next_features, close_features


def create_technical_features(df_tech, ticker: str, window: int = 240):
    rsi = df_tech[f'{ticker}_RSI14']
    sma = df_tech[f'{ticker}_SMA20']
    bb_upper = df_tech[f'{ticker}_BB_upper']
    bb_lower = df_tech[f'{ticker}_BB_lower']

    rsi_features = {f'RI{k}': rsi.shift(k) for k in range(window)[::-1]}
    sma_features = {f'SM{k}': sma.shift(k) for k in range(window)[::-1]}
    bbu_features = {f'BBU{k}': bb_upper.shift(k) for k in range(window)[::-1]}
    bbl_features = {f'BBL{k}': bb_lower.shift(k) for k in range(window)[::-1]}

    return rsi_features, sma_features, bbu_features, bbl_features
