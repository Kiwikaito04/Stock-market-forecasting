import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TRANSACTION_COST = 0.002  # 0.2%

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
                # tính PnL gộp
                pnl = df['Long'] + df['Short']

                # total turnover = |Long| + |Short|
                turnover = df['Long'].abs() + df['Short'].abs()

                # trừ phí giao dịch đúng cho cả trường hợp pnl dương hoặc âm
                df['return'] = pnl - TRANSACTION_COST * turnover

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
    labels = sorted(segment_data['label'].unique())
    color_map = get_color_map(labels)

    for label in labels:
        subset = segment_data[segment_data['label'] == label].copy()
        subset = subset.sort_values('day')
        subset['cumulative_return'] = subset['return'].cumsum()
        plt.plot(subset['day'], subset['cumulative_return'], label=label, color=color_map[label])

    plt.title(f"Phần trăm lãi tích lũy (sau phí giao dịch) - Từ {seg_start} đến {seg_end}")
    plt.xlabel("Năm")
    plt.ylabel("Lãi tích lũy (đơn vị: %)")
    plt.grid(True)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim([pd.Timestamp(f"{seg_start}-01-01"), pd.Timestamp(f"{seg_end + 1}-01-01")])

    plt.tight_layout()
    plt.show()



def plot_avg_daily_returns_bar(segment_data: pd.DataFrame, seg_start: int, seg_end: int):
    avg_returns = (
        segment_data.groupby(['label', 'year'])['return']
        .mean()
        .reset_index()
    )
    avg_returns['return'] *= 100
    pivot_df = avg_returns.pivot(index='year', columns='label', values='return')
    pivot_df = pivot_df.loc[seg_start:seg_end]

    labels = sorted(pivot_df.columns)
    color_map = get_color_map(labels)

    ax = pivot_df.plot(kind='bar', figsize=(12, 6), color=[color_map[label] for label in labels])
    plt.title(f"Lợi suất trung bình mỗi ngày theo năm - Từ {seg_start} đến {seg_end}")
    plt.xlabel("Năm")
    plt.ylabel("Lợi suất trung bình mỗi ngày")
    plt.grid(axis='y')
    plt.xticks(rotation=0)
    plt.legend(title="Mô hình")
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
    for seg_start, seg_end in segments:

        segment_data = prepare_combined_data(result_paths, seg_start, seg_end)
        segment_data = segment_data[(segment_data['year'] >= seg_start) & (segment_data['year'] <= seg_end)]

        if segment_data.empty:
            print(f"⚠️ Không có dữ liệu trong đoạn {seg_start}–{seg_end}")
            continue

        plot_cumulative_returns(segment_data, seg_start, seg_end)
        plot_avg_daily_returns_bar(segment_data, seg_start, seg_end)
        # plot_money_growth_segment(segment_data, seg_start, seg_end)

import math  # Để dùng hàm ceil
def plot_cumulative_returns_comparison_combine(result_paths: dict, segments: list[tuple[int, int]], ncols: int = 3):

    n_segments = len(segments)
    nrows = math.ceil(n_segments / ncols)
    ncols = min(n_segments, ncols)

    fig_cum, axs_cum = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4), squeeze=False)
    fig_bar, axs_bar = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4), squeeze=False)

    # 🔒 Dùng thứ tự nhãn cố định từ result_paths
    all_labels = sorted(result_paths.keys())
    color_map = get_color_map(all_labels)

    for idx, (seg_start, seg_end) in enumerate(segments):
        row = idx // ncols
        col = idx % ncols

        ax_cum = axs_cum[row][col]
        ax_bar = axs_bar[row][col]

        segment_data = prepare_combined_data(result_paths, seg_start, seg_end)
        segment_data = segment_data[(segment_data['year'] >= seg_start) & (segment_data['year'] <= seg_end)]

        if segment_data.empty:
            print(f"⚠️ Không có dữ liệu trong đoạn {seg_start}–{seg_end}")
            ax_cum.set_visible(False)
            ax_bar.set_visible(False)
            continue

        # --- Vẽ biểu đồ lãi tích lũy ---
        for label in all_labels:
            subset = segment_data[segment_data['label'] == label].copy()
            if subset.empty:
                continue
            subset = subset.sort_values('day')
            subset['cumulative_return'] = subset['return'].cumsum()
            ax_cum.plot(subset['day'], subset['cumulative_return'], label=label, color=color_map[label])

        ax_cum.set_title(f"Lãi tích lũy {seg_start}–{seg_end}")
        ax_cum.set_xlabel("Ngày")
        ax_cum.set_ylabel("Lãi tích lũy (%)")
        ax_cum.grid(True)
        ax_cum.legend()
        ax_cum.xaxis.set_major_locator(mdates.YearLocator())
        ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_cum.set_xlim([pd.Timestamp(f"{seg_start}-01-01"), pd.Timestamp(f"{seg_end + 1}-01-01")])

        # --- Vẽ biểu đồ cột lợi suất trung bình ---
        avg_returns = (
            segment_data.groupby(['label', 'year'])['return']
            .mean()
            .reset_index()
        )
        avg_returns['return'] *= 100
        pivot_df = avg_returns.pivot(index='year', columns='label', values='return')
        pivot_df = pivot_df.loc[seg_start:seg_end]

        # Chỉ giữ lại các nhãn trong all_labels để đảm bảo trật tự và màu
        pivot_df = pivot_df.reindex(columns=all_labels)

        pivot_df.plot(kind='bar', ax=ax_bar, color=[color_map[label] for label in all_labels if label in pivot_df.columns])
        ax_bar.set_title(f"Lợi suất trung bình mỗi ngày {seg_start}–{seg_end}")
        ax_bar.set_xlabel("Năm")
        ax_bar.set_ylabel("Lợi suất TB (%)")
        ax_bar.grid(axis='y')
        ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
        ax_bar.legend(title="Mô hình")
        ax_bar.set_xticklabels(pivot_df.index, rotation=0)

    # Xóa các subplot thừa nếu có
    total_plots = nrows * ncols
    for idx in range(len(segments), total_plots):
        fig_cum.delaxes(axs_cum[idx // ncols][idx % ncols])
        fig_bar.delaxes(axs_bar[idx // ncols][idx % ncols])

    fig_cum.suptitle("Biểu đồ lãi tích lũy theo đoạn", fontsize=16)
    fig_cum.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_bar.suptitle("Biểu đồ lợi suất trung bình mỗi ngày theo đoạn", fontsize=16)
    fig_bar.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

import matplotlib.colors as mcolors

def get_color_map(labels: list[str]) -> dict:
    base_colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    color_map = {}
    for i, label in enumerate(labels):
        color_map[label] = base_colors[i % len(base_colors)]
    return color_map


# ===========================
# Sử dụng:
# ===========================

RESULT_PATHS = {
    # 1990 - 2018
    # "IntraDay LSTM": "results/1993_2018-full_tickers/1993_2018-results-Intraday-240-3-LSTM",
    # "IntraDay RF": "results/1993_2018-full_tickers/1993_2018-results-Intraday-240-3-RF",
    # "NextDay LSTM": "results/1993_2018-full_tickers/1993_2018-results-NextDay-240-1-LSTM",
    # "NextDay RF": "results/1993_2018-full_tickers/1993_2018-results-NextDay-240-1-RF",

    # 2018 - 2025
    # "IntraDay LSTM": "results/2018_2025-full_tickers/2018_2025-results-Intraday-240-3-LSTM",
    # "IntraDay RF": "results/2018_2025-full_tickers/2018_2025-results-Intraday-240-3-RF",
    # "NextDay LSTM": "results/2018_2025-full_tickers/2018_2025-results-NextDay-240-1-LSTM",
    # "NextDay RF": "results/2018_2025-full_tickers/2018_2025-results-NextDay-240-1-RF",

    # So sánh đặc trưng LSTM
    "IntraDay 3-features LSTM": "results/1993_2018-full_tickers/1993_2018-results-Intraday-240-3-LSTM",
    "NextDay 1-feature LSTM": "results/1993_2018-full_tickers/1993_2018-results-NextDay-240-1-LSTM",

    # So sánh đặc trưng RF
    # "IntraDay 3-features RF": "results/1993_2018-full_tickers/1993_2018-results-Intraday-240-3-RF",
    # "NextDay 1-feature RF": "results/1993_2018-full_tickers/1993_2018-results-NextDay-240-1-RF",

}

YEAR_SEGMENTS = [
    # (1993, 2000),
    # (2001, 2009),
    # (2010, 2018),
    (1993,2018)
    # (2018, 2025)
]

plot_cumulative_returns_comparison_combine(
    result_paths=RESULT_PATHS,
    segments=YEAR_SEGMENTS,
    ncols=3
)
