"""
Model definitions and forecasting logic.
Contains implementations for various statistical, machine learning, 
and custom deep learning models used in the forecasting pipeline.
"""

import numpy as np
import pandas as pd


class SimpleMinMaxScaler:
    def __init__(self):
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None

    def fit(self, data):
        arr = np.asarray(data, dtype=float)
        self.data_min_ = arr.min(axis=0)
        self.data_max_ = arr.max(axis=0)
        denom = self.data_max_ - self.data_min_
        denom = np.where(denom == 0, 1.0, denom)
        self.scale_ = denom
        return self

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        return (arr - self.data_min_) / self.scale_

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        arr = np.asarray(data, dtype=float)
        return arr * self.scale_ + self.data_min_


def train_arima(train_series, order=(5, 1, 0)):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(train_series, order=order)
    return model.fit()


def train_sarima(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=400)


def train_prophet(train_df):
    from prophet import Prophet

    model = Prophet(daily_seasonality=True)
    model.fit(train_df)
    return model


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100.0)


def naive_forecast(last_train_value, horizon):
    return np.full(horizon, float(last_train_value), dtype=float)


def weighted_ensemble(predictions_dict, metrics_dict, epsilon=1e-6):
    valid = []
    for name, pred in predictions_dict.items():
        rmse = metrics_dict.get(name, {}).get("rmse")
        if pred is None or rmse is None or np.isnan(rmse):
            continue
        valid.append((name, np.asarray(pred, dtype=float), rmse))

    if not valid:
        return None, {}

    inv = np.array([1.0 / (row[2] + epsilon) for row in valid], dtype=float)
    weights = inv / inv.sum()
    stacked = np.vstack([row[1] for row in valid])
    ensemble = np.sum(stacked * weights[:, None], axis=0)
    weight_map = {valid[i][0]: float(weights[i]) for i in range(len(valid))}
    return ensemble, weight_map


def prepare_lstm_univariate(df, look_back=60, test_ratio=0.2):
    series = df[["Close"]].copy()
    scaler_y = SimpleMinMaxScaler()
    scaled = scaler_y.fit_transform(series)

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * (1 - test_ratio))
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, scaler_y


def prepare_lstm_multivariate(df, feature_cols, target_col="Close", look_back=60, test_ratio=0.2):
    out = df.copy().sort_values("Date").reset_index(drop=True)
    out[feature_cols] = out[feature_cols].ffill().bfill()

    scaler_x = SimpleMinMaxScaler()
    scaler_y = SimpleMinMaxScaler()

    X_scaled = scaler_x.fit_transform(out[feature_cols])
    y_scaled = scaler_y.fit_transform(out[[target_col]])

    X, y = [], []
    for i in range(look_back, len(out)):
        X.append(X_scaled[i - look_back : i, :])
        y.append(y_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * (1 - test_ratio))
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y


def train_lstm_model(X_train, y_train, input_features, epochs=5, batch_size=32):
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], input_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def inverse_lstm_preds(pred_scaled, y_scaled, scaler_y):
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    return actual, pred


def create_metrics_df(rows):
    return pd.DataFrame(
        rows,
        columns=["model", "rmse", "mae", "mape", "smape", "directional_accuracy"],
    )


def _safe_clip(pred, train_max, mult):
    arr = np.asarray(pred, dtype=float)
    return np.clip(arr, 0.01, max(train_max, 1.0) * mult)


class ARIMAStatsModel:
    def __init__(self, order=(5, 1, 2)):
        self.order = order
        self.fit_obj = None
        self.train_max = None

    def fit(self, close_values):
        from statsmodels.tsa.arima.model import ARIMA

        y = np.asarray(close_values, dtype=float)
        self.train_max = float(np.nanmax(y))
        series = pd.Series(y)
        self.fit_obj = ARIMA(series, order=self.order).fit()
        return self

    def predict(self, history, steps):
        h = np.asarray(history, dtype=float)
        if self.fit_obj is None:
            return np.full(steps, float(h[-1]), dtype=float)
        yhat = self.fit_obj.forecast(steps=steps)
        return _safe_clip(np.asarray(yhat, dtype=float), self.train_max, 30)


class SARIMAStatsModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fit_obj = None
        self.train_max = None

    def fit(self, close_values):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.asarray(close_values, dtype=float)
        self.train_max = float(np.nanmax(y))
        series = pd.Series(y)
        self.fit_obj = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=300)
        return self

    def predict(self, history, steps):
        h = np.asarray(history, dtype=float)
        if self.fit_obj is None:
            return np.full(steps, float(h[-1]), dtype=float)
        yhat = self.fit_obj.forecast(steps=steps)
        return _safe_clip(np.asarray(yhat, dtype=float), self.train_max, 50)


class ARIMALikeModel:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.ar_model = None
        self.arma_model = None
        self.last_price = None
        self.train_max = None
        self.diff_clip = 1.0

    def fit(self, close_values):
        from sklearn.linear_model import Ridge

        x = np.asarray(close_values, dtype=float)
        self.last_price = float(x[-1])
        self.train_max = float(np.nanmax(x))
        ds = np.diff(x)
        self.diff_clip = float(max(np.nanstd(ds) * 8.0, np.nanpercentile(np.abs(ds), 95), 1.0))
        if len(ds) < 20:
            self.ar_model = None
            self.arma_model = None
            return self

        # Pass 1: AR(5)
        X_ar = []
        y_ar = []
        for i in range(5, len(ds)):
            X_ar.append(ds[i - 5 : i])
            y_ar.append(ds[i])
        X_ar = np.asarray(X_ar)
        y_ar = np.asarray(y_ar)
        ar = Ridge(alpha=self.alpha)
        ar.fit(X_ar, y_ar)
        ar_pred = ar.predict(X_ar)
        resid = y_ar - ar_pred

        # Pass 2: AR(5) + MA(2)
        X_arma = []
        y_arma = []
        for i in range(7, len(ds)):
            ar_lags = ds[i - 5 : i]
            ma_lags = resid[i - 7 : i - 5]
            X_arma.append(np.concatenate([ar_lags, ma_lags]))
            y_arma.append(ds[i])
        X_arma = np.asarray(X_arma)
        y_arma = np.asarray(y_arma)
        arma = Ridge(alpha=self.alpha)
        arma.fit(X_arma, y_arma)

        self.ar_model = ar
        self.arma_model = arma
        return self

    def predict(self, history, steps):
        x = list(np.asarray(history, dtype=float))
        if len(x) < 10 or self.arma_model is None:
            return np.full(steps, float(x[-1]), dtype=float)

        ds_hist = list(np.diff(np.asarray(x, dtype=float)))
        pred_diff = []
        resid_hist = [0.0, 0.0]

        for _ in range(steps):
            ar_lags = np.asarray(ds_hist[-5:], dtype=float)
            ma_lags = np.asarray(resid_hist[-2:], dtype=float)
            ar_lags = np.nan_to_num(ar_lags, nan=0.0, posinf=self.diff_clip, neginf=-self.diff_clip)
            ma_lags = np.nan_to_num(ma_lags, nan=0.0, posinf=self.diff_clip, neginf=-self.diff_clip)
            ar_lags = np.clip(ar_lags, -self.diff_clip, self.diff_clip)
            ma_lags = np.clip(ma_lags, -self.diff_clip, self.diff_clip)
            vec = np.concatenate([ar_lags, ma_lags]).reshape(1, -1)
            d_hat = float(self.arma_model.predict(vec).flat[0])
            if not np.isfinite(d_hat):
                d_hat = 0.0
            d_hat = float(np.clip(d_hat, -self.diff_clip, self.diff_clip))
            pred_diff.append(d_hat)
            ds_hist.append(d_hat)
            resid_hist.append(0.0)

        out = np.cumsum(np.asarray(pred_diff, dtype=float)) + float(x[-1])
        return _safe_clip(out, self.train_max, 30)


class SARIMALikeModel:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.model = None
        self.last_price = None
        self.train_max = None
        self.diff_clip = 1.0

    def fit(self, close_values):
        from sklearn.linear_model import Ridge

        x = np.asarray(close_values, dtype=float)
        self.last_price = float(x[-1])
        self.train_max = float(np.nanmax(x))
        ds = np.diff(x)
        self.diff_clip = float(max(np.nanstd(ds) * 8.0, np.nanpercentile(np.abs(ds), 95), 1.0))
        if len(ds) <= 20:
            self.model = None
            return self

        sds = ds[7:] - ds[:-7]
        X = []
        y = []
        for i in range(7, len(sds)):
            X.append([sds[i - 1], sds[i - 7]])
            y.append(sds[i])
        X = np.asarray(X)
        y = np.asarray(y)
        reg = Ridge(alpha=self.alpha)
        reg.fit(X, y)
        self.model = reg
        return self

    def predict(self, history, steps):
        h = np.asarray(history, dtype=float)
        if self.model is None or len(h) < 20:
            return np.full(steps, float(h[-1]), dtype=float)

        ds = list(np.diff(h))
        sds = list((np.asarray(ds[7:]) - np.asarray(ds[:-7])).tolist())
        sds = sds if len(sds) >= 8 else [0.0] * 8

        pred_prices = []
        last_price = float(h[-1])
        for _ in range(steps):
            vec = np.asarray([sds[-1], sds[-7]], dtype=float)
            vec = np.nan_to_num(vec, nan=0.0, posinf=self.diff_clip, neginf=-self.diff_clip)
            vec = np.clip(vec, -self.diff_clip, self.diff_clip).reshape(1, -1)
            sds_hat = float(self.model.predict(vec).flat[0])
            if not np.isfinite(sds_hat):
                sds_hat = 0.0
            sds_hat = float(np.clip(sds_hat, -self.diff_clip, self.diff_clip))
            ds_hat = sds_hat + ds[-7]
            ds_hat = float(np.clip(ds_hat, -self.diff_clip, self.diff_clip))
            last_price = last_price + ds_hat
            pred_prices.append(last_price)

            sds.append(sds_hat)
            ds.append(ds_hat)

        return _safe_clip(np.asarray(pred_prices, dtype=float), self.train_max, 50)


class ProphetLikeModel:
    def __init__(self, n_changepoints=20, yearly_order=12, weekly_order=4, alpha=1.0):
        self.n_changepoints = n_changepoints
        self.yearly_order = yearly_order
        self.weekly_order = weekly_order
        self.alpha = alpha
        self.model = None
        self.mean_ = 0.0
        self.std_ = 1.0
        self.eps_ = 1e-6
        self.train_max = None

    def _features(self, n_total, n_train):
        t = np.arange(n_total, dtype=float)
        t_norm = t / max(float(n_train), 1.0)
        cols = [np.ones(n_total)]

        cps = np.linspace(0.05, 0.90, self.n_changepoints)
        for cp in cps:
            cols.append(np.maximum(t_norm - cp, 0.0))

        for k in range(1, self.yearly_order + 1):
            cols.append(np.sin(2 * np.pi * k * t / 365.25))
            cols.append(np.cos(2 * np.pi * k * t / 365.25))

        for k in range(1, self.weekly_order + 1):
            cols.append(np.sin(2 * np.pi * k * t / 7.0))
            cols.append(np.cos(2 * np.pi * k * t / 7.0))

        return np.column_stack(cols)

    def fit(self, close_values):
        from sklearn.linear_model import Ridge

        y = np.asarray(close_values, dtype=float)
        self.train_max = float(np.nanmax(y))
        y_log = np.log(y + self.eps_)
        self.mean_ = float(np.mean(y_log))
        self.std_ = float(np.std(y_log) + 1e-8)
        y_norm = (y_log - self.mean_) / self.std_

        X = self._features(n_total=len(y), n_train=len(y))
        reg = Ridge(alpha=self.alpha)
        reg.fit(X, y_norm)
        self.model = reg
        return self

    def predict(self, history, steps):
        h = np.asarray(history, dtype=float)
        if self.model is None:
            return np.full(steps, float(h[-1]), dtype=float)

        n_train = len(h)
        X_future = self._features(n_total=n_train + steps, n_train=n_train)
        yhat_norm = self.model.predict(X_future)[-steps:]
        yhat_log = yhat_norm * self.std_ + self.mean_
        yhat = np.exp(yhat_log) - self.eps_
        return _safe_clip(yhat, self.train_max, 30)


class NumpyRNNModel:
    def __init__(self, look_back=60, hidden_units=32, lr=0.002, epochs=25, batch_size=32, seed=42):
        self.look_back = look_back
        self.hidden_units = hidden_units
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.Wx = None
        self.Wh = None
        self.bh = None
        self.Wy = None
        self.by = None
        self.scaler = SimpleMinMaxScaler()
        self.train_max = None

    def _prepare(self, close_values):
        y = np.asarray(close_values, dtype=float).reshape(-1, 1)
        ys = self.scaler.fit_transform(y).ravel()
        X_seq = []
        T = []
        for i in range(self.look_back, len(ys)):
            X_seq.append(ys[i - self.look_back : i])
            T.append(ys[i])
        return np.asarray(X_seq), np.asarray(T)

    def _forward(self, seq):
        h_states = []
        h = np.zeros(self.hidden_units, dtype=float)
        for x_t in seq:
            inp = np.full(self.look_back, x_t, dtype=float)
            h = np.tanh(self.Wx @ inp + self.Wh @ h + self.bh)
            h_states.append(h)
        y_hat = float((self.Wy @ h_states[-1] + self.by).ravel()[0])
        return y_hat, h_states

    def fit(self, close_values):
        rng = np.random.default_rng(self.seed)
        X, y = self._prepare(close_values)
        if len(X) == 0:
            return self

        self.train_max = float(np.nanmax(close_values))
        self.Wx = rng.normal(0, 0.05, size=(self.hidden_units, self.look_back))
        self.Wh = rng.normal(0, 0.05, size=(self.hidden_units, self.hidden_units))
        self.bh = np.zeros(self.hidden_units, dtype=float)
        self.Wy = rng.normal(0, 0.05, size=(1, self.hidden_units))
        self.by = np.zeros(1, dtype=float)

        for _ in range(self.epochs):
            for i in range(len(X)):
                y_hat, h_states = self._forward(X[i])
                err = y_hat - y[i]
                dWy = err * h_states[-1].reshape(1, -1)
                dby = np.asarray([err], dtype=float)

                dh_next = (self.Wy.T * err).ravel()
                dWx = np.zeros_like(self.Wx)
                dWh = np.zeros_like(self.Wh)
                dbh = np.zeros_like(self.bh)

                for t in range(self.look_back - 1, -1, -1):
                    h_t = h_states[t]
                    h_prev = h_states[t - 1] if t > 0 else np.zeros(self.hidden_units, dtype=float)
                    dt = (1 - h_t ** 2) * dh_next
                    x_t = np.full(self.look_back, X[i][t], dtype=float)
                    dWx += np.outer(dt, x_t)
                    dWh += np.outer(dt, h_prev)
                    dbh += dt
                    dh_next = self.Wh.T @ dt

                for grad in [dWx, dWh, dbh, dWy, dby]:
                    np.clip(grad, -1.0, 1.0, out=grad)

                self.Wx -= self.lr * dWx
                self.Wh -= self.lr * dWh
                self.bh -= self.lr * dbh
                self.Wy -= self.lr * dWy
                self.by -= self.lr * dby
        return self

    def predict(self, history, steps):
        h = np.asarray(history, dtype=float)
        if len(h) < self.look_back or self.Wx is None:
            return np.full(steps, float(h[-1]), dtype=float)

        hs = self.scaler.transform(h.reshape(-1, 1)).ravel().tolist()
        preds_scaled = []
        for _ in range(steps):
            seq = np.asarray(hs[-self.look_back :], dtype=float)
            y_hat, _ = self._forward(seq)
            preds_scaled.append(y_hat)
            hs.append(y_hat)

        pred = self.scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        return _safe_clip(pred, self.train_max, 30)


def train_all(train_df):
    tr = np.asarray(train_df["Close"], dtype=float)
    try:
        arima = ARIMAStatsModel(order=(5, 1, 2)).fit(tr)
    except Exception:
        arima = ARIMALikeModel().fit(tr)

    try:
        sarima = SARIMAStatsModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(tr)
    except Exception:
        sarima = SARIMALikeModel().fit(tr)

    out = {
        "ARIMA": arima,
        "SARIMA": sarima,
        "Prophet-Like": ProphetLikeModel().fit(tr),
        "LSTM": NumpyRNNModel().fit(tr),
    }
    return out


def forecast_all(model_map, train_df, test_df):
    history = np.asarray(train_df["Close"], dtype=float)
    steps = len(test_df)
    preds = {}
    for name, model in model_map.items():
        preds[name] = model.predict(history, steps)
    return preds
