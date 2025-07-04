import pandas as pd
import numpy as np

def reshaper(arr):
    arr = np.array(np.split(arr, 3, axis=1))
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr

def simulate(test_data, predictions):
    rets = pd.DataFrame([], columns=['Long', 'Short'])
    k = 10
    for day in sorted(predictions.keys()):
        preds = predictions[day]
        test_returns = test_data[test_data[:, 0] == day][:, -2]
        top_preds = predictions[day].argsort()[-k:][::-1]
        trans_long = test_returns[top_preds]
        worst_preds = predictions[day].argsort()[:k][::-1]
        trans_short = -test_returns[worst_preds]
        rets.loc[day] = [np.mean(trans_long), np.mean(trans_short)]
    print('Result : ', rets.mean())
    return rets