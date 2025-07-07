import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.features import reshaper
from src.models import LSTM_Model, create_callbacks

def predictor(model, test_data):
    dates = list(set(test_data[:, 0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:, 0] == day]
        test_d = reshaper(test_d[:, 2:-2]).astype('float32')
        predictions[day] = model.predict(test_d)[:, 1]
    return predictions

# LSTM Intraday, 3 features, 240 timestep
def trainer_LSTM_I3f240(train_data, test_data, test_year, folder_save='models'):
    np.random.shuffle(train_data)

    # Các đặc trưng / Nhãn / Lợi nhuận thực tế
    train_x, train_y, train_ret = train_data[:, 2:-2], train_data[:, -1], train_data[:, -2]
    train_x = reshaper(train_x).astype('float32')
    train_y = np.reshape(train_y, (-1, 1))
    train_ret = np.reshape(train_ret, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y)
    enc_y = enc.transform(train_y).toarray()
    train_ret = np.hstack((np.zeros((len(train_data), 1)), train_ret))

    model = LSTM_Model(features=3, time_steps=240).makeLSTM()
    CALLBACK = create_callbacks(test_year, folder_save)

    model.fit(train_x, enc_y,
              epochs=1000,
              batch_size=512,
              validation_split=0.2,
              callbacks=CALLBACK,
              verbose=2)

    return model, predictor(model, test_data)