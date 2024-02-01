from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

def build_models():
    look_back = 15

    model_1 = Sequential(
        [
            LSTM(20, input_shape=(1, look_back)),
            Dense(1, activation='linear')
        ],
        name='model_1'
    )

    model_2 = Sequential(
        [
            LSTM(25, input_shape=(1, look_back), activation='relu'),
            Dense(15, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_2'
    )

    model_3 = Sequential(
        [
            LSTM(20, input_shape=(1, look_back), activation='relu'),
            Dense(12, activation='relu'),
            Dense(12, activation='relu'),
            Dense(20, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_3'
    )

    model_4 = Sequential(
        [
            LSTM(32, input_shape=(1, look_back), activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(12, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_4'
    )

    model_5 = Sequential(
        [
            LSTM(16, input_shape=(1, look_back), activation='relu', return_sequences=True),
            LSTM(8, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_5'
    )

    model_6 = Sequential(
        [
            LSTM(20, input_shape=(1, look_back), activation='tanh'),
            Dense(10, activation='relu'),
            Dense(5, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_6'
    )

    model_7 = Sequential(
        [
            LSTM(24, input_shape=(1, look_back), activation='tanh', return_sequences=True),
            LSTM(12, activation='tanh'),
            Dense(6, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_7'
    )

    model_8 = Sequential(
        [
            LSTM(30, input_shape=(1, look_back), activation='relu', return_sequences=True),
            LSTM(20, activation='relu', return_sequences=True),
            LSTM(10, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_8'
    )

    model_9 = Sequential(
        [
            LSTM(15, input_shape=(1, look_back), activation='tanh', return_sequences=True),
            LSTM(15, activation='tanh'),
            Dense(10, activation='relu'),
            Dense(5, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_9'
    )

    model_10 = Sequential(
        [
            LSTM(25, input_shape=(1, look_back), activation='relu', return_sequences=True),
            LSTM(20, activation='relu'),
            Dense(15, activation='relu'),
            Dense(5, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_10'
    )

    model_11 = Sequential(
        [
            LSTM(18, input_shape=(1, look_back), activation='relu', return_sequences=True),
            LSTM(12, activation='relu', return_sequences=True),
            LSTM(6, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_11'
    )

    model_12 = Sequential(
        [
            LSTM(30, input_shape=(1, look_back), activation='tanh', return_sequences=True),
            LSTM(20, activation='tanh', return_sequences=True),
            LSTM(10, activation='tanh'),
            Dense(1, activation='linear')
        ],
        name='model_12'
    )

    model_list = [model_1, model_2, model_3, model_4, model_5, model_6,
                  model_7, model_8, model_9, model_10, model_11, model_12]

    return model_list
