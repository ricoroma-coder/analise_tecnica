from models.charts.Chart1m import Chart1m
from models.ai.RecurrentNeuralNetworks import RNN


class Chart1mController:
    def feed_rnn_ai(self, symbol):
        chart = Chart1m(symbol)
        chart.collect_data()
        ai_model = RNN(chart.data)
        ai_model.prepare()
        ai_model.workout()
        # ai_model.generate_signals()
