import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TRANSACTION_COST = 0.0064  # 0.64%

def load_returns_from_folder(folder_path, label, start_year, end_year):
    all_returns = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(folder_path, f'daily_rets_{year}.csv')
        if not os.path.exists(file_path):
            print(f"[{label}] Không tìm thấy file: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)

            if 'Long' in df.columns and 'Short' in df.columns:
                df['return'] = (df['Long'] + df['Short']) * (1 - TRANSACTION_COST)
                df = df.reset_index().rename(columns={'index': 'day'})
                all_returns.append(df[['day', 'return']])
            else:
                print(f"[{label}] Thiếu cột Long hoặc Short trong {file_path}")
        except Exception as e:
            print(f"[{label}] Lỗi khi đọc file {file_path}: {e}")

    if not all_returns:
        return pd.DataFrame()

    full_data = pd.concat(all_returns).sort_values('day').reset_index(drop=True)
    full_data['label'] = label
    return full_data


def prepare_combined_data(result_paths: dict, start_year: int, end_year: int) -> pd.DataFrame:
    all_data = []
    for label, path in result_paths.items():
        df = load_returns_from_folder(path, label, start_year, end_year)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data)
    combined['year'] = combined['day'].dt.year
    return combined


def plot_cumulative_returns(segment_data: pd.DataFrame, seg_start: int, seg_end: int):
    plt.figure(figsize=(12, 6))
    for label in segment_data['label'].unique():
        subset = segment_data[segment_data['label'] == label].copy()
        subset = subset.sort_values('day')
        subset['cumulative_return'] = subset['return'].cumsum()
        plt.plot(subset['day'], subset['cumulative_return'], label=label)

    plt.title(f"Phần trăm lãi tích lũy (sau phí giao dịch) - Từ {seg_start} đến {seg_end}")
    plt.xlabel("Ngày")
    plt.ylabel("Lãi tích lũy (đơn vị: %)")
    plt.grid(True)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()


def plot_avg_daily_returns_bar(segment_data: pd.DataFrame, seg_start: int, seg_end: int):
    """
    Vẽ biểu đồ cột thể hiện lợi suất trung bình mỗi ngày theo từng năm.
    """
    avg_returns = (
        segment_data.groupby(['label', 'year'])['return']
        .mean()
        .reset_index()
    )

    # Chuyển lợi suất thành phần trăm
    avg_returns['return'] *= 100

    pivot_df = avg_returns.pivot(index='year', columns='label', values='return')
    pivot_df = pivot_df.loc[seg_start:seg_end]

    ax = pivot_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Lợi suất trung bình mỗi ngày theo năm - Từ {seg_start} đến {seg_end}")
    plt.xlabel("Năm")
    plt.ylabel("Lợi suất trung bình mỗi ngày")
    plt.grid(axis='y')
    plt.xticks(rotation=0)
    plt.legend(title="Mô hình")

    # Định dạng y-axis thành %
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))

    plt.tight_layout()
    plt.show()


def plot_money_growth_segment(segment_data: pd.DataFrame, seg_start: int, seg_end: int):
    """
    Vẽ biểu đồ tăng trưởng số tiền (money growth) nếu đầu tư 1 đơn vị từ đầu đoạn.
    """
    plt.figure(figsize=(12, 6))
    for label in segment_data['label'].unique():
        subset = segment_data[segment_data['label'] == label].copy()
        subset = subset.sort_values('day')

        # Đảm bảo khởi đầu từ 1
        subset['money'] = 1 * (1 + subset['return']).cumprod()
        subset.loc[subset.index[0], 'money'] = 1.0  # Reset lại điểm đầu nếu có lệch do nhân

        plt.plot(subset['day'], subset['money'], label=label)

    plt.title(f"Biểu đồ tăng trưởng tiền tích lũy (sau phí giao dịch) - Từ {seg_start} đến {seg_end}")
    plt.xlabel("Năm")
    plt.ylabel("Số tiền tích lũy (đơn vị: $)")
    plt.grid(True)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Giới hạn x-axis sát năm bắt đầu và kết thúc
    x_min = pd.Timestamp(f"{seg_start}-01-01")
    x_max = pd.Timestamp(f"{seg_end + 1}-01-01")
    ax.set_xlim([x_min, x_max])

    plt.tight_layout()
    plt.show()


def plot_cumulative_returns_comparison(result_paths: dict, segments: list[tuple[int, int]]):
    """
    Vẽ biểu đồ so sánh lãi tích lũy giữa các mô hình theo các đoạn năm chỉ định.
    Mỗi đoạn là 1 tuple (start_year, end_year).
    """
    # Lấy min/max để load đủ dữ liệu
    all_years = [y for seg in segments for y in seg]
    min_year, max_year = min(all_years), max(all_years)

    combined = prepare_combined_data(result_paths, min_year, max_year)
    if combined.empty:
        print("Không có dữ liệu nào để hiển thị.")
        return

    for seg_start, seg_end in segments:
        segment_data = combined[(combined['year'] >= seg_start) & (combined['year'] <= seg_end)]
        if not segment_data.empty:
            plot_cumulative_returns(segment_data, seg_start, seg_end)
            plot_avg_daily_returns_bar(segment_data, seg_start, seg_end)
            plot_money_growth_segment(segment_data, seg_start, seg_end)


# ===========================
# Sử dụng:
# ===========================
RESULT_PATHS = {
    "LSTM Intra 3-features": "main/ayaya/results-Intraday-240-3-LSTM",
    # "LSTM Intra 1-feature": "main/ayaya/results-Intraday-240-1-LSTM",
    "LSTM NextDay 1-feature": "main/ayaya/results-NextDay-240-1-LSTM",
    "RF Intra 3-features": "main/ayaya/results-Intraday-240-3-RF",
    # "RF Intra 1-feature": "main/ayaya/results-Intraday-240-1-RF",
    "RF NextDay 1-feature": "main/ayaya/results-NextDay-240-1-RF",
    # "LSTM 3-features": "test/result-old/report-old/results-Intraday-240-3-LSTM",
    # "LSTM 3-features with technical": "test/result-old/report-old/results-Intraday-240-3-4-LSTM",
    # "LSTM 3-features with RSI": "test/result-old/report-old/results-Intraday-240-3-1-LSTM"
}

YEAR_SEGMENTS = [
    (1993, 1999),
    (2000, 2009),
    (2010, 2018),
    (1993,2018),
]

plot_cumulative_returns_comparison(
    result_paths=RESULT_PATHS,
    segments=YEAR_SEGMENTS
)
