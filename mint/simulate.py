import pandas as pd
import numpy as np


# Đầu ra mô hình: danh sách xác xuất cổ phiếu tăng (theo thứ tự cổ phiếu ban đầu


# top_preds = [0, 2]  # Mua CP0 và CP2
# trans_long = [0.02, 0.01]
# → Long return = (0.02 + 0.01)/2 = 0.015

# worst_preds = [1, 3]  # Bán khống CP1 và CP3
# trans_short = -[-0.01, -0.03] = [0.01, 0.03]
# → Short return = (0.01 + 0.03)/2 = 0.02


# k: số lượng top cổ phiếu uptrend và downtrend
def simulate(test_data, predictions, k=10):
    rets = pd.DataFrame([], columns=['Long', 'Short'])
    for day in sorted(predictions.keys()):
        preds = predictions[day]
        test_returns = test_data[test_data[:, 0] == day][:, -2]
        top_preds = predictions[day].argsort()[-k:][::-1]
        trans_long = test_returns[top_preds]
        worst_preds = predictions[day].argsort()[:k][::-1]
        trans_short = -test_returns[worst_preds]
        rets.loc[day] = [np.mean(trans_long), np.mean(trans_short)]
    print('Result : \n', rets.mean())
    return rets