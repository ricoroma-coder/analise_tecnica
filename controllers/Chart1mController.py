from models.charts.Chart1m import Chart1m
from models.ai.RecurrentNeuralNetworks import RNN


class Chart1mController:
    def feed_rnn_ai(self, symbol):
        chart = Chart1m(symbol)
        chart.collect_data()
        ai_model = RNN(chart.data)
        print(len(chart.data))
        print(ai_model.sequence_length)
