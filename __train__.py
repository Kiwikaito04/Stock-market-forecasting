import os
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv
from mint.data import RawDataLoader
from mint.create_stock_data import create_label_LSTM_Intraday, create_stock_data_LSTM_Intraday_3f, scalar_normalize
from mint.trainer import trainer_LSTM_240

load_dotenv()
SEED = int(os.getenv("SEED", 727))
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def train_model_for_year(test_year: int, data_folder: str, model_folder: str, result_folder: str):
    START_YEAR = test_year - 3
    dataloader = RawDataLoader(data_folder, START_YEAR, test_year)
    TICKERS, df_open, df_close = dataloader.get_open_close_window(START_YEAR, test_year)

    label = create_label_LSTM_Intraday(df_open, df_close)
    train_data, test_data, tickers_in_test = [], [], []

    for ticker in TICKERS:
        try:
            st_train, st_test = create_stock_data_LSTM_Intraday_3f(df_open, df_close, label, ticker, test_year)
            train_data.append(st_train)
            test_data.append(st_test)
            tickers_in_test.append(ticker)
        except Exception as e:
            print(f"[WARNING] Skipped {ticker}: {e}")
            continue

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    scalar_normalize(train_data, test_data)

    model, probs = trainer_LSTM_240(train_data, test_data, test_year, features=3, folder_save=model_folder)
    model.save(os.path.join(model_folder, f"model-LSTM-{test_year}-final.keras"))

    # probs: shape [n_samples, 2] --> softmax probabilities
    # Extract tickers per sample (assuming in order)
    prob_df = pd.DataFrame(probs, columns=['prob_down', 'prob_up'])
    prob_df['ticker'] = np.repeat(tickers_in_test, len(test_data) // len(tickers_in_test))

    output_path = os.path.join(result_folder, f"predictions_{test_year}.csv")
    prob_df.to_csv(output_path, index=False)

    return output_path
