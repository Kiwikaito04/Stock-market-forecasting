import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import math  # Để dùng hàm ceil


import os
import pandas as pd

class ReturnDataManager:
    def __init__(self, result_paths: dict, segments: list[tuple[int, int]], transaction_cost: float = 0.002):
        self.result_paths = result_paths
        self.segments = segments
        self.transaction_cost = transaction_cost
        self.data = self._load_all_data()

    def _get_file_paths_from_segments(self, folder_path: str) -> list[str]:
        """
        Sinh danh sách các đường dẫn tới file daily_rets_{year}.csv theo đoạn năm.
        """
        file_paths = []
        for seg_start, seg_end in self.segments:
            for year in range(seg_start, seg_end + 1):
                file_path = os.path.join(folder_path, f'daily_rets_{year}.csv')
                file_paths.append((year, file_path))
        return file_paths

    def _read_and_process_file(self, label: str, year: int, file_path: str) -> pd.DataFrame | None:
        """
        Đọc 1 file daily_rets và xử lý dữ liệu trả về DataFrame đã chuẩn hóa.
        """
        if not os.path.exists(file_path):
            print(f"[{label}] Không tìm thấy file: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)

            if 'Long' not in df.columns or 'Short' not in df.columns:
                print(f"[{label}] Thiếu cột Long hoặc Short trong {file_path}")
                return None

            return self._calculate_returns_with_cost(df, label, year)

        except Exception as e:
            print(f"[{label}] Lỗi khi đọc file {file_path}: {e}")
            return None

    def _calculate_returns_with_cost(self, df: pd.DataFrame, label: str, year: int) -> pd.DataFrame:
        """
        Tính return, long_return_net, short_return_net sau khi trừ phí.
        """
        df = df[['Long', 'Short']].copy()

        long_raw = df['Long']
        short_raw = df['Short']
        turnover = long_raw.abs() + short_raw.abs()

        long_net = long_raw - self.transaction_cost * long_raw.abs()
        short_net = short_raw - self.transaction_cost * short_raw.abs()

        pnl = long_net + short_net
        total_return = pnl

        df_result = pd.DataFrame({
            'day': df.index,
            'return': total_return,
            'long_return_net': long_net,
            'short_return_net': short_net,
            'label': label,
            'year': year
        })

        return df_result.reset_index(drop=True)

    def _load_all_data(self) -> pd.DataFrame:
        """
        Tổng hợp toàn bộ dữ liệu từ các mô hình và đoạn năm.
        """
        all_data = []

        for label, folder_path in self.result_paths.items():
            file_paths = self._get_file_paths_from_segments(folder_path)

            for year, file_path in file_paths:
                df_processed = self._read_and_process_file(label, year, file_path)
                if df_processed is not None:
                    all_data.append(df_processed)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data).sort_values('day').reset_index(drop=True)

    def get_labels(self) -> list[str]:
        return sorted(self.data['label'].unique())

    def get_segment_data(self, seg: tuple[int, int]) -> pd.DataFrame:
        start, end = seg
        return self.data[(self.data['year'] >= start) & (self.data['year'] <= end)].copy()

    def get_all_data(self) -> pd.DataFrame:
        return self.data.copy()



def get_color_map(labels: list[str]) -> dict:
    base_colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    color_map = {}
    for i, label in enumerate(labels):
        color_map[label] = base_colors[i % len(base_colors)]
    return color_map


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
    plt.title(f"Lợi suất trung bình mỗi ngày - Từ {seg_start} đến {seg_end}")
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


def plot_cumulative_returns_comparison(data_manager: ReturnDataManager, segments: list[tuple[int, int]]):
    for seg_start, seg_end in segments:

        segment_data = data_manager.get_segment_data((seg_start, seg_end))
        segment_data = segment_data[(segment_data['year'] >= seg_start) & (segment_data['year'] <= seg_end)]

        if segment_data.empty:
            print(f"⚠️ Không có dữ liệu trong đoạn {seg_start}–{seg_end}")
            continue

        plot_cumulative_returns(segment_data, seg_start, seg_end)
        plot_avg_daily_returns_bar(segment_data, seg_start, seg_end)
        # plot_money_growth_segment(segment_data, seg_start, seg_end)


def plot_cumulative_returns_comparison_combine(data_manager: ReturnDataManager, segments: list[tuple[int, int]], ncols: int = 3):

    n_segments = len(segments)
    nrows = math.ceil(n_segments / ncols)
    ncols = min(n_segments, ncols)

    fig_cum, axs_cum = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4), squeeze=False)
    fig_bar, axs_bar = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4), squeeze=False)

    # 🔒 Dùng thứ tự nhãn cố định từ result_paths
    all_labels = data_manager.get_labels()
    color_map = get_color_map(all_labels)

    for idx, (seg_start, seg_end) in enumerate(segments):
        row = idx // ncols
        col = idx % ncols

        ax_cum = axs_cum[row][col]
        ax_bar = axs_bar[row][col]

        segment_data = data_manager.get_segment_data((seg_start, seg_end))
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
        ax_bar.set_title(f"Lợi suất trung bình mỗi năm {seg_start}–{seg_end}")
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
    fig_cum.tight_layout(rect=[0., 0.03, 1., 0.95])

    fig_bar.suptitle("Biểu đồ lợi suất trung bình mỗi năm theo đoạn", fontsize=16)
    fig_bar.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def plot_full_period_performance(data_manager: ReturnDataManager, start_year: int, end_year: int, time_step = 3):
    segment_data = data_manager.get_segment_data((start_year, end_year))
    segment_data = segment_data[(segment_data['year'] >= start_year) & (segment_data['year'] <= end_year)]

    if segment_data.empty:
        print(f"⚠️ Không có dữ liệu từ {start_year} đến {end_year}")
        return

    all_labels = data_manager.get_labels()
    color_map = get_color_map(all_labels)

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # --------------------------
    # 1. Biểu đồ lãi tích lũy
    # --------------------------
    ax_cum = axs[0]
    for label in all_labels:
        subset = segment_data[segment_data['label'] == label].copy()
        if subset.empty:
            continue
        subset = subset.sort_values('day')
        subset['cumulative_return'] = subset['return'].cumsum()
        ax_cum.plot(subset['day'], subset['cumulative_return'], label=label, color=color_map[label])

    ax_cum.set_title(f"Lãi tích lũy từ {start_year} đến {end_year}")
    ax_cum.set_ylabel("Lãi tích lũy (%)")
    ax_cum.grid(True)
    ax_cum.legend()

    ax_cum.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_cum.set_xlim([pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year + 1}-01-01")])
    ax_cum.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(start_year, end_year + 1 + time_step, time_step)])

    # --------------------------
    # 2. Biểu đồ lợi suất trung bình mỗi năm
    # --------------------------
    ax_bar = axs[1]

    avg_returns = (
        segment_data.groupby(['label', 'year'])['return']
        .mean()
        .reset_index()
    )
    avg_returns['return'] *= 100  # chuyển sang %
    pivot_df = avg_returns.pivot(index='year', columns='label', values='return')
    pivot_df = pivot_df.loc[start_year:end_year]
    pivot_df = pivot_df.reindex(columns=all_labels)

    pivot_df.plot(kind='bar', ax=ax_bar, color=[color_map[label] for label in all_labels])
    ax_bar.set_title("Lợi suất trung bình mỗi năm")
    ax_bar.set_xlabel("Năm")
    ax_bar.set_ylabel("Lợi suất TB (%)")
    ax_bar.grid(axis='y')
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))
    ax_bar.legend(title="Mô hình")
    ax_bar.set_xticks(range(len(pivot_df.index)))
    ax_bar.set_xticklabels(pivot_df.index, rotation=0)

    plt.tight_layout()
    plt.show()



# ===========================
# Sử dụng:
# ===========================

RESULT_PATHS = {
    # 1990 - 2018
    # "IntraDay LSTM": "results/1993_2018-full_tickers/results-Intraday-240-3-LSTM",
    # "IntraDay RF": "results/1993_2018-full_tickers/results-Intraday-240-3-RF",
    # "NextDay LSTM": "results/1993_2018-full_tickers/results-NextDay-240-1-LSTM",
    # "NextDay RF": "results/1993_2018-full_tickers/results-NextDay-240-1-RF",

    # 2018 - 2025
    # "IntraDay LSTM": "results/2018_2025-full_tickers/results-Intraday-240-3-LSTM",
    # "IntraDay RF": "results/2018_2025-full_tickers/results-Intraday-240-3-RF",
    # "NextDay LSTM": "results/2018_2025-full_tickers/results-NextDay-240-1-LSTM",
    # "NextDay RF": "results/2018_2025-full_tickers/results-NextDay-240-1-RF",

    # So sánh đặc trưng LSTM
    "IntraDay 3-features LSTM": "results/1993_2018-full_tickers/results-Intraday-240-3-LSTM",
    "NextDay 1-feature LSTM": "results/1993_2018-full_tickers/results-NextDay-240-1-LSTM",

    # So sánh đặc trưng RF
    "IntraDay 3-features RF": "results/1993_2018-full_tickers/results-Intraday-240-3-RF",
    "NextDay 1-feature RF": "results/1993_2018-full_tickers/results-NextDay-240-1-RF",

}

YEAR_SEGMENTS = [
    (1993, 2000),
    (2001, 2009),
    (2010, 2018),
    # (1993,2018)
    # (2018, 2025)
]

TRANSACTION_COST = 0.002  # 0.2%
manager = ReturnDataManager(result_paths=RESULT_PATHS, segments=YEAR_SEGMENTS, transaction_cost=TRANSACTION_COST)


# plot_cumulative_returns_comparison_combine(
#     data_manager=manager,
#     segments=YEAR_SEGMENTS,
#     ncols=3
# )

plot_full_period_performance(manager, start_year=1993, end_year=2018)
