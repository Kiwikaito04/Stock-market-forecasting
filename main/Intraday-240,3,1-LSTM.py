import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random

from src.data import fetch_yfinance_data
from src.create_stock_data import create_label_LSTM_Intraday, create_stock_data_LSTM_Intraday_3f_technical, scalar_normalize, reshaper
from src.simulate import simulate
from src.Statistics import Statistics
from src.trainer import trainer_LSTM_I3f240
from ticker_list import get_ticker_name


# DATA CONFIGURATION
TICKERS = get_ticker_name()
START_YEAR, END_YEAR = 1993, 2018
WINDOW_SIZE = 4


# CHECKPOINT CONFIGURATION
MODEL_FOLDER = 'report-old/models-Intraday-240-3-1-LSTM'
RESULT_FOLDER = 'report-old/results-Intraday-240-3-1-LSTM'
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# RANDOM SEED SETUP
SEED = 727
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)


for test_year in range(START_YEAR + WINDOW_SIZE, END_YEAR + 1):
    print(f"\n{'='*20} Testing {test_year} {'='*20}\n")

    start_date = f"{test_year - WINDOW_SIZE}-01-01"
    end_date = f"{test_year}-12-31"

    df_open, df_close, df_tech = fetch_yfinance_data(TICKERS, start_date, end_date, True)
    label = create_label_LSTM_Intraday(df_open, df_close)

    train_data, test_data = [], []
    for ticker in TICKERS:
        try:
            st_train, st_test = create_stock_data_LSTM_Intraday_3f_technical(df_open, df_close, df_tech, label, ticker, test_year)
            train_data.append(st_train)
            test_data.append(st_test)
        except Exception as e:
            print(f"Skipped {ticker}: {e}")
            continue

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scalar_normalize(train_data, test_data)

    # Huấn luyện mô hình và đưa ra dự đoán, kiểm tra kết quả
    model, predictions = trainer_LSTM_I3f240(train_data, test_data, test_year, features=4, folder_save=MODEL_FOLDER)
    returns = simulate(test_data, predictions, k=10)
    returns.to_csv(f"{RESULT_FOLDER}/daily_rets_{test_year}.csv")

    stats = Statistics(returns.sum(axis=1))
    stats.shortreport()

    # report = generate_prediction_report(test_data, model)
    # report.to_csv(f"{RESULT_FOLDER}/daily_report_{test_year}.csv")
    with open(RESULT_FOLDER + "/avg_returns.txt", "a") as myfile:
        res = '-' * 30 + '\n'
        res += str(test_year) + '\n'
        res += 'Mean = ' + str(stats.mean()) + '\n'
        res += 'Sharpe = ' + str(stats.sharpe()) + '\n'
        res += '-' * 30 + '\n'
        myfile.write(res)
