import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random
import warnings
warnings.filterwarnings("ignore")

from src.data import fetch_yfinance_data
from src.create_stock_data import create_label_LSTM_Intraday, create_stock_data_LSTM_Intraday_3f, scalar_normalize, reshaper
from src.simulate import simulate
from src.Statistics import Statistics
from src.trainer import trainer_LSTM_240
from src.utils import get_ticker_name


# DATA CONFIGURATION
TICKERS = get_ticker_name()
START_YEAR, END_YEAR = 1990, 2018
WINDOW_SIZE = 3


# CHECKPOINT CONFIGURATION
MODEL_FOLDER = 'ayaya/models-Intraday-240-3-LSTM-Attention'
RESULT_FOLDER = 'ayaya/results-Intraday-240-3-LSTM-Attention'
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# RANDOM SEED SETUP
SEED = 727
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)


summary_rows = []


for test_year in range(START_YEAR + WINDOW_SIZE, END_YEAR + 1):
    print(f"\n{'='*20} Testing {test_year} {'='*20}\n")

    start_date = f"{test_year - WINDOW_SIZE}-01-01"
    end_date = f"{test_year}-12-31"

    df_open, df_close = fetch_yfinance_data(TICKERS, start_date, end_date)
    label = create_label_LSTM_Intraday(df_open, df_close)

    train_data, test_data = [], []
    for ticker in TICKERS:
        try:
            st_train, st_test = create_stock_data_LSTM_Intraday_3f(df_open, df_close, label, ticker, test_year)
            train_data.append(st_train)
            test_data.append(st_test)
        except Exception as e:
            print(f"Skipped {ticker}: {e}")
            continue

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scalar_normalize(train_data, test_data)

    # Huấn luyện mô hình và đưa ra dự đoán, kiểm tra kết quả
    model, predictions = trainer_LSTM_240(train_data, test_data, test_year, features=3, folder_save=MODEL_FOLDER, use_attention=True)
    returns = simulate(test_data, predictions, k=10)
    returns.to_csv(f"{RESULT_FOLDER}/daily_rets_{test_year}.csv")

    stats = Statistics(returns.sum(axis=1))
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
    model.save(os.path.join(MODEL_FOLDER, f"model-LSTM-{test_year}-final.keras"))


summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RESULT_FOLDER, "summary_stats.csv"), index=False)

