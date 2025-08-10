import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from mint.data import RawDataLoader
from mint.create_stock_data import (
    create_label_LSTM_Intraday,
    create_stock_data_LSTM_Intraday_3f,
    scalar_normalize,
)
from mint.create_stock_data import reshaper
from tensorflow.keras.models import load_model


load_dotenv()
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 3))
DATA_FOLDER = os.getenv("DATA_FOLDER", "dataset")
MODEL_FOLDER = os.getenv("APP_MODELS", "app_src/models")
RESULT_FOLDER = os.getenv("APP_RESULTS", "app_src/results")


def _prepare_window_data_for_predict(test_year: int):
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
        raise ValueError("No samples available for prediction.")

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    scalar_normalize(train_data, test_data)
    return test_data


def _predict_year_from_model(model_path: str, test_year: int):
    model = load_model(model_path)
    test_data = _prepare_window_data_for_predict(test_year)

    # Build predictions per day as in your pipeline
    dates = list(set(test_data[:, 0]))
    predictions = {}
    for day in dates:
        mask = test_data[:, 0] == day
        rows = test_data[mask]
        X = reshaper(rows[:, 2:-2]).astype('float32')  # keep shape identical to trainer
        preds = model.predict(X, verbose=0)[:, 1]      # prob of "up" class
        predictions[day] = preds

    # Save one CSV with all (Date, Ticker, ProbUp)
    records = []
    for day in dates:
        rows = test_data[test_data[:, 0] == day]
        tickers = rows[:, 1]
        probs = predictions[day]
        for tkr, p in zip(tickers, probs):
            records.append((day, tkr, float(p)))

    df = pd.DataFrame(records, columns=["Date", "Ticker", "ProbUp"])
    df.sort_values(["Date", "ProbUp"], ascending=[True, False], inplace=True)
    out_path = os.path.join(RESULT_FOLDER, f"preds_{test_year}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def predict_year(test_year: int, model_basename: str | None = None):
    """
    Predict using an already-trained model file.
    If model_basename is None, tries 'model-LSTM-{test_year}-final.keras'.
    Returns path to preds_{test_year}.csv
    """
    if model_basename is None:
        model_basename = f"model-LSTM-{test_year}-final.keras"
    model_path = os.path.join(MODEL_FOLDER, model_basename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return _predict_year_from_model(model_path, test_year)
