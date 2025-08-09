import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from mint.data import RawDataLoader
from mint.create_stock_data import create_label_LSTM_Intraday, create_stock_data_LSTM_Intraday_3f, scalar_normalize
from mint.trainer import predict_with_model  # bạn cần định nghĩa hàm này

def predict_with_existing_model(test_year: int, data_folder: str, model_folder: str, result_folder: str):
    dataloader = RawDataLoader(data_folder, test_year - 3, test_year)
    TICKERS, df_open, df_close = dataloader.get_open_close_window(test_year - 3 , test_year)

    label = create_label_LSTM_Intraday(df_open, df_close)
    test_data, tickers_in_test = [], []

    for ticker in TICKERS:
        try:
            _, st_test = create_stock_data_LSTM_Intraday_3f(df_open, df_close, label, ticker, test_year)
            test_data.append(st_test)
            tickers_in_test.append(ticker)
        except Exception as e:
            print(f"[WARNING] Skipped {ticker}: {e}")
            continue

    test_data = np.concatenate(test_data)
    scalar_normalize(test_data, test_data)  # for inference, just normalize

    model_path = os.path.join(model_folder, f"model-LSTM-{test_year}-final.keras")
    model = load_model(model_path)
    probs = predict_with_model(model, test_data)

    prob_df = pd.DataFrame(probs, columns=['prob_down', 'prob_up'])
    prob_df['ticker'] = np.repeat(tickers_in_test, len(test_data) // len(tickers_in_test))

    output_path = os.path.join(result_folder, f"predictions_{test_year}.csv")
    prob_df.to_csv(output_path, index=False)

    return output_path
