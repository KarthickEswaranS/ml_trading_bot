from src.features import Features 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


class TrainTest(Features):

    def __init__(self):
        super().__init__()

    def train_test(self):
        df = self.feature()
        indicators = ['ret_1','ret_3','ma_5','ma_10','ma_20','std_10','mom_5','rsi']
        X = df[indicators].values
        y = df['target'].values

        tscv = TimeSeriesSplit(n_splits=5)
        # print(tscv)

        # We'll perform a simple train-test split: last 20% as test
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        dates_train = df.index[:split_index]
        dates_test = df.index[split_index:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    

        # MODEL TRAINING
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        print('=================Final Result===============')
        print("Test accuracy:", accuracy_score(y_test, y_pred))
        print('-----------------Test Report-----------------')
        print(classification_report(y_test, y_pred))

        # Feature importances
        importances = pd.Series(model.feature_importances_, index=indicators).sort_values(ascending=False)
        print('-----------------Feature importances-----------------')
        print("Feature importances:\n", importances)
        print("Feature importances:\n", importances.idxmax())

        return y_pred,y_proba
    
# tt = TrainTest()
# tt.train_test()