import os
import pandas as pd
import matplotlib.pyplot as plt


def export_metric_table(summary_paths: dict, metric_map: dict[str, str]) -> pd.DataFrame:
    """
    Xuất bảng các giá trị trung bình của các thước đo, theo từng mô hình.
    - metric_map: dict[tên hiển thị -> tên cột trong summary_stats.csv]
    """
    result = {}

    for model_name, folder_path in summary_paths.items():
        csv_path = os.path.join(folder_path, 'summary_stats.csv')
        if not os.path.exists(csv_path):
            print(f"⚠️ Không tìm thấy file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        model_metrics = {}

        for display_name, col_name in metric_map.items():
            if col_name not in df.columns:
                print(f"⚠️ Không có cột '{col_name}' trong {csv_path}")
                continue

            value = df[col_name].mean()
            model_metrics[display_name] = value

        result[model_name] = model_metrics

    if not result:
        raise ValueError("⚠️ Không có dữ liệu nào để xuất bảng.")

    df_result = pd.DataFrame(result)
    df_result.index.name = "Metric"
    return df_result


from cumulative_returns_calc import ReturnDataManager
def export_long_short_mean(result_paths: dict, segments: list[tuple[int, int]]) -> pd.DataFrame:
    """
    Tính trung bình Long và Short return cho mỗi mô hình trên toàn bộ các năm.
    """
    manager = ReturnDataManager(result_paths, segments)

    df = manager.get_all_data()
    if df.empty:
        raise ValueError("Không có dữ liệu để tính long/short return.")

    grouped = df.groupby('label')[['long_return_net', 'short_return_net']].mean() * 100  # nhân 100 để đưa về %
    grouped = grouped.rename(columns={
        'long_return': 'Mean (Long) (%)',
        'short_return': 'Mean (Short) (%)'
    })

    return grouped.T  # transpose để giống bảng export_metric_table


def plot_bar_metric_across_models(
    metric: str,
    summary_paths: dict,
    segments: list[tuple[int, int]],
    transaction_cost: float = 0.002,
    ordered_labels: list[str] = None,
    metric_name_map: dict[str, str] = None
):
    """
    Vẽ biểu đồ cột cho một chỉ số đã chọn, theo thứ tự mô hình mong muốn.
    """

    # Map tên chỉ số hiển thị
    if metric_name_map is None:
        metric_name_map = {
            'sharpe': 'Sharpe Ratio',
            'mean': 'Mean Return (%)',
            'std': 'Standard Deviation',
            'long_return_net': 'Mean Long Return (%)',
            'short_return_net': 'Mean Short Return (%)',
            'return': 'Mean Net Return (%)',
        }

    display_name = metric_name_map.get(metric, metric)

    # Lấy dữ liệu theo loại chỉ số
    if metric in ['mean', 'std', 'sharpe', 'stderr', 'pos_perc', 'skewness', 'kurtosis', 'min', 'max']:
        df_metrics = export_metric_table(summary_paths, {display_name: metric})
        df_plot = df_metrics.loc[display_name]

    elif metric in ['long_return_net', 'short_return_net', 'return']:
        manager = ReturnDataManager(summary_paths, segments, transaction_cost)
        df = manager.get_all_data()
        if df.empty:
            raise ValueError("Không có dữ liệu để vẽ.")
        df_plot = df.groupby('label')[metric].mean() * 100  # nhân 100 để thành %

    else:
        raise ValueError(f"Không hỗ trợ metric: {metric}")

    # Đảm bảo đúng thứ tự mô hình
    if ordered_labels is None:
        ordered_labels = list(summary_paths.keys())

    df_plot = df_plot.reindex(ordered_labels)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    df_plot.plot(kind='bar', color='steelblue')
    plt.xticks(rotation=0, ha='center')  # chữ nằm ngang
    plt.ylabel(display_name)
    plt.title(display_name)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



RESULT_PATHS = {
    # So sánh đặc trưng LSTM
    "IntraDay 3-features LSTM": "results/1993_2018-full_tickers/results-Intraday-240-3-LSTM",
    "NextDay 1-feature LSTM": "results/1993_2018-full_tickers/results-NextDay-240-1-LSTM",

    # So sánh đặc trưng RF
    "IntraDay 3-features RF": "results/1993_2018-full_tickers/results-Intraday-240-3-RF",
    "NextDay 1-feature RF": "results/1993_2018-full_tickers/results-NextDay-240-1-RF",

    "IntraDay 1-features LSTM": "main-old/full-30_tickers/ayaya/results-IntraDay-240-1-LSTM",
    "IntraDay 1-features RF": "main-old/full-30_tickers/ayaya/results-IntraDay-240-1-RF",
}

df = export_long_short_mean(RESULT_PATHS, [(1993, 2018)])
print(df)
df.to_csv("result-long_short.csv")


SUMMARY_PATHS  = {
    "IntraDay 3-features LSTM": "results/1993_2018-full_tickers/results-Intraday-240-3-LSTM",
    "IntraDay 3-features RF": "results/1993_2018-full_tickers/results-Intraday-240-3-RF",

    "IntraDay 1-features LSTM": "main-old/full-30_tickers/ayaya/results-IntraDay-240-1-LSTM",
    "IntraDay 1-features RF": "main-old/full-30_tickers/ayaya/results-IntraDay-240-1-RF",

    "NextDay 1-features LSTM": "results/1993_2018-full_tickers/results-NextDay-240-1-LSTM",
    "NextDay 1-features RF": "results/1993_2018-full_tickers/results-NextDay-240-1-RF",

}


metric_map = {
    "Mean Return (%)": "mean",
    "Standard Deviation": "std",
    "Sharpe Ratio": "sharpe",
    "Standard Error": "stderr",
    "Share>0": "pos_perc",
    "Skewness": "skewness",
    "Kurtosis": "kurtosis",
    "Min": "min",
    "Max": "max",
}

df_metrics = export_metric_table(SUMMARY_PATHS, metric_map)
print(df_metrics)
df_metrics.to_csv("result.csv")


YEAR_SEGMENTS = [
    (1993, 2018),
]
metric_name_map = {

    'sharpe': 'Sharpe Ratio',
    'pos_perc': 'Share>0',
    'mean': 'Mean Return (%)',
    'long_return_net': 'Mean Long Return (%)',
    'short_return_net': 'Mean Short Return (%)',
}


plot_bar_metric_across_models(
    metric='sharpe',
    summary_paths=SUMMARY_PATHS,
    segments=YEAR_SEGMENTS,
    ordered_labels=[
        "IntraDay 3-features LSTM",
        "NextDay 1-features LSTM",
        "IntraDay 3-features RF",
        "NextDay 1-features RF",
        "IntraDay 1-features LSTM",
        "IntraDay 1-features RF",
    ],
    metric_name_map=metric_name_map
)

