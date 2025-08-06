import os
import pickle
import numpy as np
import random
import pandas as pd
from dotenv import load_dotenv

from mint.data import RawDataLoader
from mint.create_stock_data import create_label_NextDay, create_stock_data_RF_NextDay_1f
from mint.simulate import simulate
from mint.Statistics import Statistics
from mint.trainer import trainer_RF


# DATA CONFIGURATION
load_dotenv()
START_YEAR = int(os.getenv("START_YEAR", 1990))
END_YEAR = int(os.getenv("END_YEAR", 2018))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 3))
DATA_FOLDER = os.getenv("DATA_FOLDER", "dataset")
os.makedirs(DATA_FOLDER, exist_ok=True)
SEED = int(os.getenv("SEED", 727))


# CHECKPOINT CONFIGURATION
MODEL_FOLDER = 'ayaya/models-NextDay-240-1-RF'
RESULT_FOLDER = 'ayaya/results-NextDay-240-1-RF'

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# RANDOM SEED SETUP
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)


dataloader = RawDataLoader(DATA_FOLDER, START_YEAR, END_YEAR)
summary_rows = []


# ======================== CHẠY QUA TỪNG NĂM ========================
for test_year in range(START_YEAR + WINDOW_SIZE, END_YEAR + 1):
    print(f"\n{'='*20} Testing {test_year} {'='*20}\n")

    # Lấy dữ liệu
    print("[DEBUG] Khởi tạo tập dữ liệu...")
    TICKERS, _, df_close = dataloader.get_open_close_window(test_year - WINDOW_SIZE, test_year)
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
    print("[DEBUG] Hoàn tất.")

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
