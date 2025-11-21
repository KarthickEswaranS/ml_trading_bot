from src.backtest import Backtest
import numpy as np
import mplfinance as mplf

class Plot(Backtest):
        
    def __init__(self):
        super().__init__()

    def plot(self):
        df = self.backtest()
        # Create "long signals" for mplfinance
        df['Long_Signal'] = np.where(df['y_pred'] == 1, df['Close'], np.nan)

        # Select last 200 candles
        # n = 200
        # subset = df.tail(n)

        # Addplot for long signals
        long_ap = mplf.make_addplot(
            df['Long_Signal'],
            type='scatter',
            marker='^',
            markersize=100,
            color='brown'
        )

        # Plot with mplfinance
        mplf.plot(
            df,
            type='candle',
            style='yahoo',
            addplot=[long_ap],
            title=f"ML Predicted LONG Signals)",
            volume=True,
            figsize=(14, 7)
        )
        print(df)
        # plt.figure(figsize=(12,6))
        # plt.plot(df.index, df['cum_strategy'], label='Strategy')
        # plt.plot(df.index, df['cum_buy_and_hold'], label='Buy & Hold', alpha=0.7)
        # plt.legend()
        # plt.title("Cumulative Returns - Strategy vs Buy & Hold")
        # plt.xlabel("Date")
        # plt.ylabel("Cumulative Return")
        # plt.show()

        # Plot signals on price chart (last 200 bars)
        # n = 200
        # subset = df.tail(n)
        # plt.figure(figsize=(14,6))
        # plt.plot(subset.index, subset['Close'], label='Close Price')
        # longs = subset[subset['y_pred'] == 1]
        # plt.scatter(longs.index, longs['Close'], marker='^', color='green', label='Predicted LONG', s=50)
        # plt.legend()
        # plt.title("Price with Predicted Long Signals (last {} bars)".format(n))
        # plt.show()

p = Plot()
p.train_test()
p.backtest()
p.plot()
