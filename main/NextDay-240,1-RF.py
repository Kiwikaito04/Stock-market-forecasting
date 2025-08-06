import os
import pickle
import numpy as np
import random

import pandas as pd

from mint.data import fetch_yfinance_data
from mint.create_stock_data import create_label_NextDay, create_stock_data_RF_NextDay_1f
from mint.simulate import simulate
from mint.Statistics import Statistics
from mint.trainer import trainer_RF
from mint.utils import get_ticker_name, get_valid_tickers


# DATA CONFIGURATION
TICKERS = get_ticker_name()
START_YEAR, END_YEAR = 1990, 2018
WINDOW_SIZE = 3

# CHECKPOINT CONFIGURATION
MODEL_FOLDER = 'ayaya/models-NextDay-240-1-RF'
RESULT_FOLDER = 'ayaya/results-NextDay-240-1-RF'
DATA_FOLDER = 'dataset'

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)


SEED = 727
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ======================== TẢI DỮ LIỆU 1 LẦN ========================
data_open_path = os.path.join(DATA_FOLDER, "df_open.csv")
data_close_path = os.path.join(DATA_FOLDER, "df_close.csv")
valid_tickers_path = os.path.join(DATA_FOLDER, "valid_tickers.csv")

if os.path.exists(data_open_path) and os.path.exists(data_close_path):
    df_open_all = pd.read_csv(data_open_path)
    df_close_all = pd.read_csv(data_close_path)
    TICKERS = get_valid_tickers(valid_tickers_path)
    print("[INFO] Đã tải dữ liệu từ file CSV.")
else:
    TICKERS, df_open_all, df_close_all = fetch_yfinance_data(TICKERS, f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    # df_open_all.to_csv(data_open_path, index=False)
    df_close_all.to_csv(data_close_path, index=False)
    pd.Series(TICKERS).to_csv(valid_tickers_path, index=False, header=False)
    print("[INFO] Đã lưu dữ liệu vào CSV.")


summary_rows = []


# ======================== CHẠY QUA TỪNG NĂM ========================
for test_year in range(START_YEAR + WINDOW_SIZE, END_YEAR + 1):
    print(f"\n{'='*20} Testing {test_year} {'='*20}\n")

    start_date = f"{test_year - WINDOW_SIZE}-01-01"
    end_date = f"{test_year}-12-31"

    # Lọc dữ liệu theo ngày
    mask = (df_close_all['Date'] >= start_date) & (df_close_all['Date'] <= end_date)
    # df_open = df_open_all[mask].reset_index(drop=True)
    df_close = df_close_all[mask].reset_index(drop=True)

    label = create_label_NextDay(df_close)

    train_data, test_data = [], []
    for ticker in TICKERS:
        try:
            st_train, st_test = create_stock_data_RF_NextDay_1f(df_close, label, ticker, test_year)
            train_data.append(st_train)
            test_data.append(st_test)
        except Exception as e:
            print(f"[WARNING]: Skipped {ticker}: {e}")
            continue

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    model, predictions = trainer_RF(train_data, test_data, SEED=SEED)
    returns = simulate(test_data, predictions)
    returns.to_csv(f"{RESULT_FOLDER}/daily_rets_{test_year}.csv")

    stats = Statistics(returns.sum(axis=1))
    print('\nAverage returns prior to transaction charges')
    stats.shortreport()

    summary_rows.append({
        'year': test_year,
        'mean': stats.mean(),
        'std': stats.std(),
        'sharpe': stats.sharpe(),
        'stderr': stats.stderr(),
        'pos_perc': stats.pos_perc(),
        'skewness': stats.skewness(),
        'kurtosis': stats.kurtosis(),
        'VaR_1': stats.VaR(1),
        'VaR_2': stats.VaR(2),
        'VaR_5': stats.VaR(5),
        'CVaR_1': stats.CVaR(1),
        'CVaR_2': stats.CVaR(2),
        'CVaR_5': stats.CVaR(5),
        'MDD': stats.MDD(),
        **stats.percentiles()
    })

    # Save model with specific name
    model_path = os.path.join(MODEL_FOLDER, f"model-RF-{test_year}-final.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RESULT_FOLDER, "summary_stats.csv"), index=False)
