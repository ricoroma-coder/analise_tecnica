import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.layers.core import Flatten, Dense
from tensorflow.python.layers.pooling import MaxPooling1D
from models.ai.AIModel import AIModel


class CNN(AIModel):  # Convolutionals Neural Networks
    def create_sequences(self, data):
        x, y = [], []

        for i in range(len(data) - self.sequence_length):
            x.append(data.iloc[i:(i + self.sequence_length)].values)
            y.append(data.iloc[i + self.sequence_length].values)

        x_cnn = np.expand_dims(x, axis=-1)
        super().create_sequences((x_cnn, y))

    def workout(self, adjust_params=True):
        self.load_model('cnn.h5')
        adjust = False

        if self.model is None:
            adjust = True
            self.model = Sequential()
            self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                                  input_shape=(self.sequence_length, len(self.selected_features), 1)))
            self.model.add(MaxPooling1D(pool_size=2))
            self.model.add(Flatten())
            self.model.add(Dense(len(self.selected_features)))
            self.model.compile(optimizer='adam', loss='mean_squared_error')

        super().workout(adjust_params=adjust)
