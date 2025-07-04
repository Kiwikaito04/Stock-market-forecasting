import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers

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
                                 monitor='val_loss', save_best_only=False)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    return [csv_logger, early_stop, checkpoint]