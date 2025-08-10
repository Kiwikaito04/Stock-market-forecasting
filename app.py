import os
import pandas as pd
import streamlit as st
from datetime import date

from __train__ import train_year
from __predict__ import predict_year


MODEL_FOLDER = os.getenv("APP_MODELS", "app_src/models")
RESULT_FOLDER = os.getenv("APP_RESULTS", "app_src/results")
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


MIN_YEAR = 1990
MAX_YEAR = 2100
DEFAULT_VALUE = 2025


##########################
#       Thiết lập
##########################

# Thiết lập title
st.set_page_config(page_title="LSTM Intraday Predictor", layout="wide")
st.title("📈 LSTM Intraday – Train & Predict")


# Thiết lập các lựa chọn
mode = st.radio("Chế độ", ["Predict", "Train"], horizontal=True)
col1, col2 = st.columns(2)
with col1:
    k = st.number_input("Chọn k (Top k tăng/giảm)", min_value=1, max_value=100, value=10, step=1)
with col2:
    pick_date = st.date_input("Chọn ngày dự đoán", value=date.today())

test_year = pick_date.year


st.caption("Lưu ý: Train dùng rolling window 3 năm trước test_year (ví dụ test_year=2025 → train 2022–2024).")


# Chọn model pre-trained
model_basename = None
if mode == "Predict":
    model_basename = st.text_input("Model filename (tùy chọn)", value=f"model-LSTM-{test_year}-final.keras")


run = st.button("Run")


##########################
#   Cài đặt function
##########################
def _load_or_build_predictions_csv(mode, test_year, model_basename):
    preds_path = os.path.join(RESULT_FOLDER, f"preds_{test_year}.csv")
    if os.path.exists(preds_path):
        st.info(f"Đã tìm thấy file dự đoán: {preds_path}")
        return preds_path

    with st.spinner("Đang tạo dự đoán cho năm đã chọn..."):
        if mode == "Train":
            model_path, preds_path = train_year(test_year=test_year, features=3)
            return preds_path
        else:
            preds_path = predict_year(
                test_year=test_year,
                model_basename=model_basename if model_basename else None
            )
            return preds_path

def _topk_for_date(preds_df: pd.DataFrame, target_date: str, k: int):
    day_df = preds_df[preds_df["Date"] == target_date]
    if day_df.empty:
        return None, None
    up = day_df.sort_values("ProbUp", ascending=False).head(k)[["Ticker", "ProbUp"]].reset_index(drop=True)
    down = day_df.sort_values("ProbUp", ascending=True).head(k)[["Ticker", "ProbUp"]].reset_index(drop=True)
    return up, down


##########################
#       Chạy
##########################
if run:
    preds_path = _load_or_build_predictions_csv(mode, test_year, model_basename)
    df = pd.read_csv(preds_path)

    target_date = pick_date.strftime("%Y-%m-%d")
    up_k, down_k = _topk_for_date(df, target_date, k)

    if up_k is None:
        st.warning(f"⚠️ Không có dự đoán cho ngày {target_date}. Hãy chọn ngày khác trong năm {test_year}.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Top {k} tăng – {target_date}")
            st.dataframe(up_k)
        with c2:
            st.subheader(f"Top {k} giảm – {target_date}")
            st.dataframe(down_k)

        # Save the selected-day outputs (as requested)
        up_path = os.path.join(RESULT_FOLDER, f"top_{k}_up_{target_date}.csv")
        down_path = os.path.join(RESULT_FOLDER, f"top_{k}_down_{target_date}.csv")
        up_k.to_csv(up_path, index=False)
        down_k.to_csv(down_path, index=False)
        st.success("Đã lưu CSV top-k cho ngày đã chọn.")
        st.write("📄", up_path)
        st.write("📄", down_path)
