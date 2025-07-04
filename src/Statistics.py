import pandas as pd
import numpy as np
import scipy.stats


class Statistics:
    def __init__(self, series):
        self.series = np.array(series)
        self.n = len(series)

    # Lợi suất trung bình
    def mean(self):
        return np.mean(self.series)

    # Độ lệch chuẩn (biến động / rủi ro)
    def std(self):
        return np.std(self.series)

    # Sai số chuẩn (standard error of the mean)
    # Độ tin cậy của ước lượng trung bình
    def stderr(self):
        return scipy.stats.sem(self.series)

    # Thống kê mô tả: min, max, 25%, 50% (median), 75%, ...
    def percentiles(self, p=[.25, .5, .75]):
        return pd.Series(self.series).describe(percentiles=p)

    # Tỷ lệ phần trăm ngày có lợi suất dương (>0)
    # Tần suất lời
    def pos_perc(self):
        return 100 * sum(self.series > 0) / self.n

    # Độ lệch phân phối
    # >0 -> thiên lệch phải (nhiều lợi nhuận lớn bất thường)
    # <0 -> thiên lệch trái (nhiều lỗ to bất thường)
    def skewness(self):
        return scipy.stats.skew(self.series)

    #  Độ nhọn phân phối
    # Cao → nhiều giá trị cực đoan (fat tails)
    def kurtosis(self):
        return scipy.stats.kurtosis(self.series)

    # Ví dụ: VaR(5) = lợi suất tệ nhất trong top 5%
    # Ý nghĩa: 95% xác suất là bạn không thua quá mức này
    def VaR(self, confidence):
        indx = int(confidence * self.n / 100)
        return sorted(self.series)[indx - 1]

    #  Giá trị kỳ vọng khi thua lỗ tệ nhất (Expected Shortfall)
    def CVaR(self, confidence):
        indx = int(confidence * self.n / 100)
        return sum(sorted(self.series)[:indx]) / indx

    # Giả lập tài sản tăng theo chuỗi lợi suất
    # Tìm điểm rơi sâu nhất từ đỉnh cao nhất → đo mức lỗ lớn nhất
    def MDD(self):
        money = np.cumprod(1 + self.series / 100)
        maximums = np.maximum.accumulate(money)
        drawdowns = 1 - money / maximums
        return np.max(drawdowns)

    # → Sharpe Ratio (điều chỉnh rủi ro):
    # Đo lợi suất vượt trội so với lãi suất phi rủi ro
    # 252: số ngày giao dịch trong năm → annual hóa
    # So sánh hiệu suất dựa trên rủi ro
    def sharpe(self, risk_free_rate=0.0003):
        mu = self.mean()
        sig = self.std()
        sharpe_d = (mu - risk_free_rate) / sig
        return (252 ** 0.5) * sharpe_d

    # In báo cáo
    def shortreport(self):
        print('Mean \t\t', self.mean())
        print('Standard dev \t', self.std())
        print('Sharpe ratio \t', self.sharpe())

    # In báo cáo chi tiết
    def report(self):
        print('Mean \t\t', self.mean())
        print('Standard dev \t', self.std())
        print('Sharpe ratio \t', self.sharpe())
        print('Standard Error \t', self.stderr())
        print('Share>0 \t', self.pos_perc())
        print('Skewness \t', self.skewness())
        print('Kurtosis \t', self.kurtosis())
        print('VaR_1 \t\t', self.VaR(1))
        print('VaR_2 \t\t', self.VaR(2))
        print('VaR_5 \t\t', self.VaR(5))
        print('CVaR_1 \t\t', self.CVaR(1))
        print('CVaR_2 \t\t', self.CVaR(2))
        print('CVaR_5 \t\t', self.CVaR(5))
        print('MDD \t\t', self.MDD())
        print(self.percentiles())
