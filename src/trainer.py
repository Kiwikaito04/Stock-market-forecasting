import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.create_stock_data import reshaper

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers


##############
#       LSTM
##############

# =================== Model LSTM =================== #
class LSTM_Model:
    def __init__(self, features=3, time_steps=240):
        self.inputs = Input(shape=(time_steps, features))
        x = LSTM(25, return_sequences=False)(self.inputs)
        x = Dropout(0.1)(x)
        self.outputs = Dense(2, activation='softmax')(x)

    def makeLSTM(self):
        model = Model(inputs=self.inputs, outputs=self.outputs)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(),
                      metrics=['accuracy'])
        model.summary()
        return model


def create_callbacks(test_year, model_type='LSTM', folder='models'):
    csv_logger = CSVLogger(f"{folder}/training-log-{model_type}-{test_year}.csv")
    checkpoint = ModelCheckpoint(f"{folder}/model-{model_type}-{test_year}-E{{epoch:02d}}.keras",
                                 monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    return [csv_logger, early_stop, checkpoint]

# =================== Trainer LSTM =================== #
def predictor_LSTM(model, test_data, features):
    dates = list(set(test_data[:, 0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:, 0] == day]
        test_d = reshaper(test_d[:, 2:-2], features=features).astype('float32')
        predictions[day] = model.predict(test_d)[:, 1]
    return predictions


# LSTM Intraday, 3 features, 240 timestep
def trainer_LSTM_240(train_data, test_data, test_year, features=3, folder_save='models'):
    np.random.shuffle(train_data)

    # Các đặc trưng / Nhãn / Lợi nhuận thực tế
    train_x, train_y, train_ret = train_data[:, 2:-2], train_data[:, -1], train_data[:, -2]
    train_x = reshaper(train_x, features=features).astype('float32')
    train_y = np.reshape(train_y, (-1, 1))
    train_ret = np.reshape(train_ret, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y)
    enc_y = enc.transform(train_y).toarray()
    train_ret = np.hstack((np.zeros((len(train_data), 1)), train_ret))

    model = LSTM_Model(features=features, time_steps=240).makeLSTM()
    CALLBACK = create_callbacks(test_year, folder=folder_save)

    model.fit(train_x, enc_y,
              epochs=1000,
              batch_size=512,
              validation_split=0.2,
              callbacks=CALLBACK,
              verbose=2)

    return model, predictor_LSTM(model, test_data, features)


##############
#       RF
##############

# =================== Model & Trainer RF =================== #
def predictor_RF(model, test_data):
    dates = list(set(test_data[:, 0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:, 0] == day]
        test_d = test_d[:, 2:-2]
        predictions[day] = model.predict_proba(test_d)[:, 1]
    return predictions


def trainer_RF(train_data, test_data, MAX_DEPTH=10, SEED=42):
    train_x, train_y = train_data[:, 2:-2], train_data[:, -1]
    train_y = train_y.astype('int')

    print('Started training')
    clf = RandomForestClassifier(n_estimators=1000,
                                 max_depth=MAX_DEPTH,
                                 random_state=SEED,
                                 n_jobs=-1)
    clf.fit(train_x, train_y)
    print('Completed ', clf.score(train_x, train_y))

    return clf, predictor_RF(clf, test_data)