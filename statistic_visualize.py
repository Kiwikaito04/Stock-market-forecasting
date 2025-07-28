import os
import pandas as pd
import matplotlib.pyplot as plt

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_summary_data(summary_paths: dict, metric_cols: list) -> pd.DataFrame:
    """
    Đọc các file summary_stats.csv từ các thư mục và gom lại thành một DataFrame.
    """
    all_data = []

    for model_name, folder_path in summary_paths.items():
        csv_path = os.path.join(folder_path, 'summary_stats.csv')
        if not os.path.exists(csv_path):
            print(f"⚠️ File không tồn tại: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        required = set(['year'] + metric_cols)
        if not required.issubset(df.columns):
            print(f"⚠️ File thiếu cột cần thiết trong {csv_path}")
            continue

        df = df[['year'] + metric_cols].copy()
        df['Model'] = model_name
        all_data.append(df)

    if not all_data:
        raise ValueError("⚠️ Không có dữ liệu nào được nạp.")

    df_all = pd.concat(all_data)
    df_all['year'] = df_all['year'].astype(int)
    return df_all


def filter_by_year_segment(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Trích dữ liệu theo đoạn năm.
    """
    return df[(df['year'] >= start) & (df['year'] <= end)]


def plot_metric_by_year(df: pd.DataFrame, metric: str, year_segment: tuple, title_prefix: str = '', transaction_cost: float = 0.0):
    """
    Vẽ biểu đồ cột trung bình theo từng năm cho một chỉ số cụ thể.
    Có thể áp dụng điều chỉnh theo transaction_cost nếu metric là 'mean'.
    """
    seg_start, seg_end = year_segment
    df_segment = filter_by_year_segment(df, seg_start, seg_end)

    if df_segment.empty:
        print(f"⚠️ Không có dữ liệu cho đoạn {seg_start}-{seg_end}")
        return

    df_plot = df_segment.copy()

    # Điều chỉnh mean nếu cần trừ phí giao dịch
    if metric == 'mean' and transaction_cost > 0:
        df_plot[metric] -= transaction_cost * 100  # giả định phí ở dạng %, mean cũng ở dạng %

    pivot_df = df_plot.pivot_table(
        index='year',
        columns='Model',
        values=metric,
        aggfunc='mean'
    ).sort_index()

    # Vẽ
    ax = pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'{title_prefix} {metric.capitalize()} per Year ({seg_start}–{seg_end})')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Year')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()


def plot_avg_sharpe_by_year(summary_paths: dict, year_segments: list):
    """
    Hàm chính: Vẽ Sharpe ratio trung bình theo năm, chia đoạn.
    """
    df = load_summary_data(summary_paths, metric_cols=['sharpe'])

    for segment in year_segments:
        plot_metric_by_year(df, metric='sharpe', year_segment=segment, title_prefix='Average')


def plot_avg_mean_by_year(summary_paths: dict, year_segments: list, transaction_cost: float = 0.0):
    """
    Hàm phụ: Vẽ Mean return trung bình theo năm, có thể điều chỉnh phí giao dịch.
    """
    df = load_summary_data(summary_paths, metric_cols=['mean'])

    for segment in year_segments:
        plot_metric_by_year(df, metric='mean', year_segment=segment, title_prefix='Average', transaction_cost=transaction_cost)





SUMMARY_PATHS  = {
    "IntraDay LSTM": "results/results-Intraday-240-3-LSTM",
    "IntraDay RF": "results/results-Intraday-240-3-RF",
    "NextDay LSTM": "results/results-NextDay-240-1-LSTM",
    "NextDay RF": "results/results-NextDay-240-1-RF",
}


YEAR_SEGMENTS = [
    (1993, 2000),
    (2001, 2009),
    (2010, 2018),
]

# Vẽ Sharpe
plot_avg_sharpe_by_year(SUMMARY_PATHS, YEAR_SEGMENTS)
