import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from tensorflow.python.keras.models import load_model


class AIModel:
    model = None
    data = []
    selected_features = []
    sequence_length = 0
    test, train = [{'x': [], 'y': []} for _ in range(2)]

    def __init__(self, data):
        self.data = data

    def load_model(self, model_name):
        model_dir = '../../temp/models'
        if os.path.exists(model_dir):
            self.model = load_model(os.path.join(model_dir, model_name))

    def prepare(self):
        self.selected_features = ['High', 'Low', 'Close', 'Volume']
        self.data = self.data[self.selected_features]

        data = self.data.ffill()
        data['Daily_Return'] = data['Close'].pct_change()

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        columns = self.selected_features + ['Daily_Return']
        data_normalized = pd.DataFrame(data_scaled, columns=columns, index=data.index)

        self.create_sequences(data_normalized)

    def create_sequences(self, data):
        self.set_sequences(data)

    def set_sequences(self, data):
        x, y = data
        self.train['x'], self.test['x'], self.train['y'], self.test['y'] = train_test_split(
            np.array(x), np.array(y), test_size=0.2, train_size=0.8, shuffle=False)

    def workout(self, adjust_params=True):
        self.model.fit(self.train['x'], self.train['y'], epochs=10, batch_size=32, validation_data=(
            self.test['x'], self.test['y']))

        if adjust_params:
            self.adjust_hyperparams()

    def adjust_hyperparams(self):
        grid = {'epochs': [5, 10, 15], 'batch_size': [32, 64, 128]}
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid, scoring='neg_mean_squared_error',
                                   cv=TimeSeriesSplit(n_splits=3))
        grid_result = grid_search.fit(self.train['x'], self.train['y'])
        best_params = grid_result.best_params_

        #  henrique
        print('MSE: ', self.get_mse(best_params))

    def get_mse(self, params):
        self.model.fit(self.train['x'], self.train['y'], epochs=params['epochs'], batch_size=params['batch_size'])
        return self.model.evaluate(self.test['x'], self.test['x'])

    def generate_signals(self):
        predictions = self.model.predict(self.test['x'])
        signals = np.where(predictions > 0.5, 'Comprar', 'Vender')
        signals[np.abs(predictions - 0.5) <= 0.1] = 'Manter'

        #  henrique
        print('Signals: ', signals)
