from src.train_test import TrainTest
import numpy as np

class Backtest(TrainTest):
       
    def __init__(self):
        super().__init__()

    def backtest(self):
        df = self.feature()
        split_index = int(len(df) * 0.8)
        y_pred,y_proba = self.train_test()
    

        test_df = df.iloc[split_index:].copy()
        test_df['y_pred'] = y_pred
        test_df['y_proba'] = y_proba
        test_df['next_open'] = test_df['Open'].shift(-1) 
        test_df['strategy_ret'] = 0.0
        test_df['strategy_ret'] = test_df['target_ret'] * test_df['y_pred']

        # Cumulative returns
        test_df['cum_strategy'] = (1 + test_df['strategy_ret']).cumprod()
        test_df['cum_buy_and_hold'] = (1 + test_df['target_ret']).cumprod()
    
        ann_sharpe = self.sharpe(test_df['strategy_ret'].values, period=252)
        ann_sharpe_bh = self.sharpe(test_df['target_ret'].values, period=252)
        total_return = test_df['cum_strategy'].iloc[-1] - 1
        bh_return = test_df['cum_buy_and_hold'].iloc[-1] - 1

        print('-----------------Sharpe Ratio-----------------')
        print(f"Strategy total return: {total_return:.2%}, \nSharpe (ann): {ann_sharpe:.2f}")
        print(f"Buy&Hold return: {bh_return:.2%}, \nSharpe (ann): {ann_sharpe_bh:.2f}")

        # self.plot(test_df)
        return test_df

    def sharpe(self, returns, period=252):
        # returns: series of periodic returns
        mean = np.nanmean(returns)
        std = np.nanstd(returns)
        if std == 0:
            return np.nan
        return (mean / std) * np.sqrt(period)
    
# b = Backtest()
# b.backtest()