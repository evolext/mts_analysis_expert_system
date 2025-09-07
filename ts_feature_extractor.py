import numpy as np
import pandas as pd

from pycatch22 import catch22_all
from scipy.signal import argrelextrema
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


DEFAULT_PERIODS = [4, 7, 12, 24, 48, 52, 96, 144, 168, 336, 672, 1008, 1440]


class TimeSeriesFeatureExtractorParallel:

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs

    def read_data(self, path: str, nrows=None) -> pd.DataFrame:
        data = pd.read_csv(path)
        label_exists = "label" in data["cols"].values
        all_points = data.shape[0]
        columns = data.columns

        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()

        is_univariate = n_points == all_points
        n_cols = all_points // n_points
        df = pd.DataFrame()
        cols_name = data["cols"].unique()

        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points: (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        else:
            df[cols_name[0]] = data.iloc[:, 0]

        if label_exists:
            last_col_name = df.columns[-1]
            df.rename(columns={last_col_name: "label"}, inplace=True)
            df = df.drop(columns='label')

        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]

        return df

    def fft_transfer(self, timeseries: np.ndarray, fmin: float = 0.2):
        yf = abs(np.fft.fft(timeseries))
        yfnorm = yf / len(timeseries)
        yfhalf = yfnorm[:len(timeseries) // 2] * 2
        fwbest = yfhalf[argrelextrema(yfhalf, np.greater)]
        xwbest = argrelextrema(yfhalf, np.greater)
        fwbest = fwbest[fwbest >= fmin].copy()
        return len(timeseries) / xwbest[0][:len(fwbest)], fwbest

    def adjust_period(self, p):
        for std in DEFAULT_PERIODS:
            if abs(p - std) <= max(1, 0.05 * std):
                return std
        return p

    def process_column(self, col_name, series):
        try:
            adf_p = adfuller(series.values, autolag="AIC")[1]
        except:
            adf_p = np.nan

        try:
            periods, amplitudes = self.fft_transfer(series.values)
            sorted_amps = sorted(amplitudes, reverse=True)
            periods_list = []
            for amp in sorted_amps:
                idx = amplitudes.tolist().index(amp)
                p = int(round(periods[idx]))
                p = self.adjust_period(p)
                if p >= 4 and p not in periods_list:
                    periods_list.append(p)
        except:
            periods_list = []

        final_periods = periods_list[:3] + DEFAULT_PERIODS
        unique_periods = list({p for p in final_periods if p >= 4})

        yuzhi = max(len(series) // 3, 12)
        seasonality_candidates = {}
        for period in unique_periods:
            if period < yuzhi:
                try:
                    res = STL(series, period=period).fit()
                    orig, trend, seasonal, resid = series, res.trend, res.seasonal, res.resid
                    detrend = orig - trend
                    deseasonal = orig - seasonal
                    trend_strength = 0 if deseasonal.var() == 0 else max(0, 1 - resid.var() / deseasonal.var())
                    seasonal_strength = 0 if detrend.var() == 0 else max(0, 1 - resid.var() / detrend.var())
                    seasonality_candidates[seasonal_strength] = (seasonal_strength, trend_strength)
                except:
                    continue

        if len(seasonality_candidates) < 3:
            for i in range(3 - len(seasonality_candidates)):
                seasonality_candidates[0.1 * (i + 1)] = (-1, -1)

        sorted_strengths = sorted(seasonality_candidates.items(), key=lambda x: x[0], reverse=True)
        top_strengths = sorted_strengths[:3]
        seasonal_strength1 = top_strengths[0][1][0]
        trend_strength1 = top_strengths[0][1][1]

        try:
            catch = catch22_all(series.values)
            transition = catch["values"][catch["names"].index("SB_TransitionMatrix_3ac_sumdiagcov")]
            shifting = catch["values"][catch["names"].index("DN_OutlierInclude_p_001_mdrmd")]
            features = np.array(catch["values"])
        except:
            transition = np.nan
            shifting = np.nan
            features = np.full(22, np.nan)

        return {
            "Variable": col_name,
            "Transition": transition,
            "Shifting": shifting,
            "Seasonality": seasonal_strength1,
            "Trend": trend_strength1,
            "Stationarity": adf_p
        }, features

    def feature_extract(self, path: str) -> pd.DataFrame:
        df = self.read_data(path)

        with tqdm_joblib(tqdm(total=len(df.columns), desc="Processing columns")):        
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_column)(col, df[col].dropna()) for col in df.columns
            )
            
        rows, feature_vectors = zip(*results)
        df_result = pd.DataFrame(rows)

        try:
            embeddings = np.stack(feature_vectors)
            scaler = MinMaxScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)

            n = embeddings_scaled.shape[0]
            corr_list = [abs(np.corrcoef(embeddings_scaled[i], embeddings_scaled[j])[0, 1])
                         for i in range(n) for j in range(i + 1, n)]
            if corr_list:
                mean_corr = np.mean(corr_list)
                var_corr = np.var(corr_list)
                correlation = 2 * (mean_corr + 1 / (var_corr + 2)) / 3
            else:
                correlation = np.nan
        except:
            correlation = np.nan

        df_result["Correlation"] = correlation
        return df_result

