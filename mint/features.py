def create_price_features(df_open, df_close, ticker: str, window: int = 240):
    daily_change = df_close[ticker] / df_open[ticker] - 1
    nextday_ret = df_open[ticker].shift(-1) / df_close[ticker] - 1
    close_change = df_close[ticker].pct_change(fill_method=None)

    intra_features = {f'IntraR{k}': daily_change.shift(k) for k in range(window)[::-1]}
    next_features = {f'NextR{k}': nextday_ret.shift(k) for k in range(window)[::-1]}
    close_features = {f'CloseR{k}': close_change.shift(k) for k in range(window)[::-1]}

    return intra_features, next_features, close_features
