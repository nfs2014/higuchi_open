from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def normalize_windows(window_data):
    normalized_data = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    for window in window_data:#51個のデータを標準化
        normalized = scaler.fit_transform(window[:,3:])
        normalized_window = np.concatenate([window[:,:3],normalized],axis=1)
        #ラベル以外を標準化
        normalized_data.append(normalized_window)
    return normalized_data#(712,51,12)

