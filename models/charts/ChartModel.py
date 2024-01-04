import yfinance as yf


class ChartModel:
    period = ''
    interval = ''
    symbol = ''
    data = []

    def __init__(self, symbol):
        self.symbol = symbol

    def collect_data(self):
        if self.period:
            self.data = yf.download(self.symbol, period=self.period, interval=self.interval)

            self.data.to_csv(f'temp/csv/{self.symbol}_{self.period}.csv')  # temporary
