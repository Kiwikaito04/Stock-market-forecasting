import pandas as pd

def get_ticker_name():
    file_path = '../data/stock-name.csv'  # Đường dẫn mặc định đến file CSV

    try:
        df = pd.read_csv(file_path)
        tickers = df['Symbol'].dropna().unique().tolist()
        tickers = [t.replace('.', '-') for t in tickers] # Chuẩn hóa tên ticker
        return tickers
    except Exception as e:
        print(f"[ERROR] get_ticker_name(): Lỗi khi đọc file ticker: {e}")
        return []


def get_valid_tickers(path="valid_tickers.csv"):
    try:
        tickers = pd.read_csv(path, header=None).squeeze().dropna().unique().tolist()
        tickers = [t.replace('.', '-') for t in tickers]  # Chuẩn hóa tên ticker (nếu cần)
        return tickers
    except Exception as e:
        print(f"[ERROR] get_valid_tickers(): {e}")
        return []


def get_default_ticker_name():
    return [
        'AAPL',   # Apple - Công nghệ
        'MSFT',   # Microsoft - Công nghệ
        'INTC',   # Intel - Bán dẫn
        'IBM',    # IBM - Công nghệ
        'CSCO',   # Cisco Systems - Mạng & phần cứng
        'ORCL',   # Oracle - Phần mềm
        'QCOM',   # Qualcomm - Bán dẫn

        'JNJ',    # Johnson & Johnson - Chăm sóc sức khỏe
        'PFE',    # Pfizer - Dược phẩm
        'MRK',    # Merck - Dược phẩm
        'BMY',    # Bristol-Myers Squibb - Dược phẩm
        'LLY',    # Eli Lilly - Dược phẩm

        'JPM',    # JPMorgan Chase - Ngân hàng
        'BAC',    # Bank of America - Ngân hàng
        'WFC',    # Wells Fargo - Ngân hàng
        'C',      # Citigroup - Ngân hàng
        'MS',     # Morgan Stanley - Dịch vụ tài chính

        'XOM',    # Exxon Mobil - Năng lượng
        'CVX',    # Chevron - Năng lượng
        'SLB',    # Schlumberger - Dịch vụ dầu khí

        'KO',     # Coca-Cola - Đồ uống
        'PEP',    # PepsiCo - Đồ uống
        'PG',     # Procter & Gamble - Tiêu dùng thiết yếu
        'CL',     # Colgate-Palmolive - Hàng tiêu dùng
        'WMT',    # Walmart - Bán lẻ

        'MCD',    # McDonald's - Dịch vụ ăn uống
        'DIS',    # Disney - Giải trí
        'T',      # AT&T - Viễn thông
        'VZ',     # Verizon - Viễn thông
        'GE'      # General Electric - Công nghiệp
    ]