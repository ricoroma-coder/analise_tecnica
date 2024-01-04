from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.layers.core import Dense
from models.ai.AIModel import AIModel


class LSTMM(AIModel):  # Long Short Term Memory Model
    def create_sequences(self, data):
        x, y = [], []

        for i in range(len(data) - self.sequence_length):
            x.append(data.iloc[i:(i + self.sequence_length)].values)
            y.append(data.iloc[i + self.sequence_length].values)

        super().create_sequences((x, y))

    def workout(self, adjust_params=True):
        self.load_model('lstm.h5')
        adjust = False

        if self.model is None:
            adjust = True
            self.model = Sequential()
            self.model.add(
                LSTM(50, activation='relu', input_shape=(self.sequence_length, len(self.selected_features))))
            self.model.add(Dense(len(self.selected_features)))
            self.model.compile(optimizer='adam', loss='mean_squared_error')

        super().workout(adjust_params=adjust)
