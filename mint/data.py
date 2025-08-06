import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class YfinanceLoader():
    def __init__(self, valid_tickers='../dataset/valid_tickers.csv' , tickers_path='../dataset/stock-name.csv', tickers=None):
        if not tickers:
            self.tickers = self._get_ticker_name(valid_tickers)
        if not self.tickers:
            self.tickers = self._get_ticker_name(tickers_path)


    @staticmethod
    def _get_ticker_name(tickers_path):
        try:
            df = pd.read_csv(tickers_path)
            tickers = df['Symbol'].dropna().unique().tolist()
            tickers = [t.replace('.', '-') for t in tickers]  # Chuẩn hóa tên ticker
            return tickers
        except Exception as e:
            print(f"[ERROR] _get_ticker_name(): Lỗi khi đọc file ticker: {e}")
            return []

        
    def fetch_yfinance_data(self,start_date, end_date, buffer=30):
        # Tính ngày bắt đầu có thêm buffer
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=buffer)
        start_buffer = start_dt.strftime("%Y-%m-%d")

        data = yf.download(
            tickers=' '.join(self.tickers),
            start=start_buffer,
            end=end_date,
            progress=False,
            auto_adjust=True,
            group_by='ticker',
            threads=True
        )

        open_list, close_list = [], []
        valid_tickers = []

        for ticker in self.tickers:
            ticker_data = data.get(ticker)
            if ticker_data is None:
                continue
            open_series = ticker_data['Open']
            close_series = ticker_data['Close']

            # Nếu toàn bộ dữ liệu là NaN thì bỏ qua
            if open_series.isna().all() or close_series.isna().all():
                continue

            valid_tickers.append(ticker)
            open_list.append(open_series.rename(ticker))
            close_list.append(close_series.rename(ticker))

        print(f"[INFO] fetch_yfinance_data(): Lấy thành công dữ liệu cho {len(valid_tickers)}/{len(self.tickers)} mã.")

        if not valid_tickers:
            return pd.DataFrame(), pd.DataFrame()

        # Gộp các series lại
        df_open = pd.concat(open_list, axis=1)
        df_close = pd.concat(close_list, axis=1)

        # Chuyển index thành Date
        df_open.insert(0, 'Date', df_open.index.strftime('%Y-%m-%d'))
        df_close.insert(0, 'Date', df_close.index.strftime('%Y-%m-%d'))

        # Cắt bỏ buffer
        mask = df_close['Date'] >= start_date
        df_open = df_open[mask].reset_index(drop=True)
        df_close = df_close[mask].reset_index(drop=True)

        return valid_tickers, df_open, df_close


    @staticmethod
    def get_default_ticker_name():
        return [
            'AAPL',  # Apple - Công nghệ
            'MSFT',  # Microsoft - Công nghệ
            'INTC',  # Intel - Bán dẫn
            'IBM',  # IBM - Công nghệ
            'CSCO',  # Cisco Systems - Mạng & phần cứng
            'ORCL',  # Oracle - Phần mềm
            'QCOM',  # Qualcomm - Bán dẫn

            'JNJ',  # Johnson & Johnson - Chăm sóc sức khỏe
            'PFE',  # Pfizer - Dược phẩm
            'MRK',  # Merck - Dược phẩm
            'BMY',  # Bristol-Myers Squibb - Dược phẩm
            'LLY',  # Eli Lilly - Dược phẩm

            'JPM',  # JPMorgan Chase - Ngân hàng
            'BAC',  # Bank of America - Ngân hàng
            'WFC',  # Wells Fargo - Ngân hàng
            'C',  # Citigroup - Ngân hàng
            'MS',  # Morgan Stanley - Dịch vụ tài chính

            'XOM',  # Exxon Mobil - Năng lượng
            'CVX',  # Chevron - Năng lượng
            'SLB',  # Schlumberger - Dịch vụ dầu khí

            'KO',  # Coca-Cola - Đồ uống
            'PEP',  # PepsiCo - Đồ uống
            'PG',  # Procter & Gamble - Tiêu dùng thiết yếu
            'CL',  # Colgate-Palmolive - Hàng tiêu dùng
            'WMT',  # Walmart - Bán lẻ

            'MCD',  # McDonald's - Dịch vụ ăn uống
            'DIS',  # Disney - Giải trí
            'T',  # AT&T - Viễn thông
            'VZ',  # Verizon - Viễn thông
            'GE'  # General Electric - Công nghiệp
        ]


class DataLoader:
    def __init__(self, data_folder, start_year, end_year):
        self.data_folder = data_folder
        self._load_data(start_year, end_year)


    def _load_data(self, start_year, end_year):
        data_open_path = os.path.join(self.data_folder, "df_open.csv")
        data_close_path = os.path.join(self.data_folder, "df_close.csv")
        valid_tickers_path = os.path.join(self.data_folder, "valid_tickers.csv")

        if not os.path.exists(data_open_path) or not os.path.exists(data_close_path):
            valid_tickers, df_open_all, df_close_all = YfinanceLoader().fetch_yfinance_data(
                f"{start_year}-01-01",
                f"{end_year}-12-31")
            df_open_all.to_csv(data_open_path, index=False)
            df_close_all.to_csv(data_close_path, index=False)
            pd.Series(valid_tickers).to_csv(valid_tickers_path, index=False, header=False)
            print("[INFO] Đã lưu dữ liệu vào CSV.")

        print("[INFO] Dữ liệu đã sẵn sàng!")


    def _get_valid_tickers(self):
        path = os.path.join(self.data_folder, f"valid_tickers.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} không tồn tại.")
        return pd.read_csv(path, header=None)[0].tolist()

    def _load_price_data(self, kind: str) -> pd.DataFrame:

        assert kind in ('open', 'close'), "kind phải là 'open' hoặc 'close'"

        path = os.path.join(self.data_folder, f"df_{kind}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} không tồn tại.")
        df = pd.read_csv(path)
        return df


    def _get_price_window(self, kind: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._load_price_data(kind)
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        return df.loc[mask].reset_index(drop=True)

    def get_open_close_window(self, start_year: int, end_year: int, tickers: list[str] = None
            ) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        # Lấy danh sách tickers
        if tickers is None:
            tickers = self._get_valid_tickers()

        df_open = self._get_price_window('open', start_date, end_date)
        df_close = self._get_price_window('close', start_date, end_date)

        # Chỉ giữ các cột tương ứng ticker mong muốn
        cols = ['Date'] + [t for t in tickers if t in df_open.columns]

        df_open = df_open[cols]
        df_close = df_close[cols]

        return tickers, df_open.reset_index(drop=True), df_close.reset_index(drop=True)