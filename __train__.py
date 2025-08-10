import os
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# === Your modules (unchanged usage) ===
from mint.data import RawDataLoader
from mint.create_stock_data import (
    create_label_LSTM_Intraday,
    create_stock_data_LSTM_Intraday_3f,
    scalar_normalize,
)
from mint.trainer import trainer_LSTM_240
# predictions is expected to be: dict[date_str] -> np.array(prob_up) aligned with per-day ticker order


# ---------- Config ----------
load_dotenv()
SEED = int(os.getenv("SEED", 727))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 3))
DATA_FOLDER = os.getenv("DATA_FOLDER", "dataset")
MODEL_FOLDER = os.getenv("APP_MODELS", "app_src/models")
RESULT_FOLDER = os.getenv("APP_RESULTS", "app_src/results")
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ---------- Repro ----------
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def _prepare_window_data(test_year: int):
    """
    Build train_data / test_data for a single test_year using your exact pipeline.
    Returns:
        TICKERS: list[str]
        df_open, df_close: DataFrames
        train_data, test_data: np.ndarray
    """

    dataloader = RawDataLoader(DATA_FOLDER, test_year - WINDOW_SIZE, test_year, mode_get_new=True)
    TICKERS, df_open, df_close = dataloader.get_open_close_window(test_year - WINDOW_SIZE, test_year)
    label = create_label_LSTM_Intraday(df_open, df_close)

    train_data, test_data = [], []
    for ticker in TICKERS:
        try:
            st_train, st_test = create_stock_data_LSTM_Intraday_3f(
                df_open, df_close, label, ticker, test_year
            )
            train_data.append(st_train)
            test_data.append(st_test)
        except Exception as e:
            print(f"[WARNING] Skipped {ticker}: {e}")

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Found array with 0 sample(s) after building window data.")

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    scalar_normalize(train_data, test_data)
    return TICKERS, df_open, df_close, train_data, test_data


def _extract_day_frames(test_data: np.ndarray):
    """
    Map each date -> (tickers_in_order, row_indices)
    Assumes your test_data has the first two columns [Date, Name] string-like.
    """
    dates = list(set(test_data[:, 0]))
    per_day = {}
    for day in dates:
        mask = test_data[:, 0] == day
        rows = test_data[mask]
        tickers = rows[:, 1].tolist()   # Name column
        idxs = np.where(mask)[0]
        per_day[day] = (tickers, idxs)
    return per_day


def _save_year_predictions_csv(test_year: int, test_data: np.ndarray, probs_by_day: dict):
    """
    Save one CSV with ALL per-ticker probabilities for the test_year:
      columns: Date,Ticker,ProbUp
    """
    records = []
    per_day = _extract_day_frames(test_data)
    for day, (tickers, _) in per_day.items():
        probs = probs_by_day.get(day)
        if probs is None:
            continue
        for tkr, p in zip(tickers, probs):
            records.append((day, tkr, float(p)))

    df = pd.DataFrame(records, columns=["Date", "Ticker", "ProbUp"])
    df.sort_values(["Date", "ProbUp"], ascending=[True, False], inplace=True)
    out_path = os.path.join(RESULT_FOLDER, f"preds_{test_year}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def train_year(test_year: int, features: int = 3, window_size: int = 3):
    """
    Train on [test_year - window_size, ..., test_year - 1], test on test_year.
    Saves:
      - model keras: model-LSTM-{test_year}-final.keras
      - predictions CSV: preds_{test_year}.csv with per-ticker ProbUp for each Date.
    Returns model_path, preds_csv_path
    """
    print(f"\n======== Training for test_year={test_year} ========")
    TICKERS, df_open, df_close, train_data, test_data = _prepare_window_data(test_year, window_size)

    model, predictions = trainer_LSTM_240(
        train_data, test_data, test_year, features=features, folder_save=MODEL_FOLDER
    )
    # Save model
    model_path = os.path.join(MODEL_FOLDER, f"model-LSTM-{test_year}-final.keras")
    model.save(model_path)

    # Save predictions (per-date probability arrays)
    preds_csv_path = _save_year_predictions_csv(test_year, test_data, predictions)
    print(f"[INFO] Saved model -> {model_path}")
    print(f"[INFO] Saved predictions -> {preds_csv_path}")
    return model_path, preds_csv_path
