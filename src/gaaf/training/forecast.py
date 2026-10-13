import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import joblib

class ForecastFeatureEngineer:
    """
    Extracts time-based and lag-based features suitable for edge tabular models.
    """
    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.set_index("timestamp")

        # Temporal features
        hour = data.index.hour + data.index.minute / 60.0
        dayofweek = data.index.dayofweek

        # Cyclical encoding for diurnal continuity
        data["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        data["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
        data["day_of_week"] = dayofweek
        data["is_weekend"] = (dayofweek >= 5).astype(int)

        # Lag features (previous 15-min step, 1-hour step, and 24-hour cycle)
        for col in ["p_load_kw", "p_pv_kw"]:
            if col in data.columns:
                data[f"{col}_lag1"] = data[col].shift(1)    # 15 min ago
                data[f"{col}_lag4"] = data[col].shift(4)    # 1 hour ago
                data[f"{col}_lag96"] = data[col].shift(96)  # 24 hours ago (same time yesterday)

        return data


class ForecastAgent:
    """
    Edge-resident Forecast Agent predicting load and solar generation
    over the next sliding window horizon.
    """
    def __init__(self):
        # Edge-optimized hyperparameters: low tree count and shallow depth for <5ms inference
        self.load_model = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )
        self.pv_model = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )
        self.load_feature_cols = [
            "sin_hour", "cos_hour", "day_of_week", "is_weekend",
            "p_load_kw_lag1", "p_load_kw_lag4", "p_load_kw_lag96"
        ]
        self.pv_feature_cols = [
            "sin_hour", "cos_hour", "day_of_week", "is_weekend",
            "p_pv_kw_lag1", "p_pv_kw_lag4", "p_pv_kw_lag96"
        ]
        self.is_trained = False

    def train(self, df_raw: pd.DataFrame):
        """
        Trains both Load and PV models on historical telemetry.
        """
        data = ForecastFeatureEngineer.extract_features(df_raw).dropna()

        # Train Load Forecaster
        X_load = data[self.load_feature_cols]
        y_load = data["p_load_kw"]
        self.load_model.fit(X_load, y_load)

        # Train PV Forecaster
        X_pv = data[self.pv_feature_cols]
        y_pv = data["p_pv_kw"]
        self.pv_model.fit(X_pv, y_pv)

        self.is_trained = True

    def predict(self, current_timestamp: pd.Timestamp, recent_history: pd.DataFrame) -> dict:
        """
        Predicts p_load and p_pv for t+1 (next 15 minutes).
        recent_history: DataFrame containing at least the past 96 steps (24 hours).
        """
        if not self.is_trained:
            raise RuntimeError("Forecast Agent models must be trained before inference.")

        # Construct single-row feature vector for next timestamp
        hour = current_timestamp.hour + current_timestamp.minute / 60.0
        dow = current_timestamp.dayofweek

        feat_dict = {
            "sin_hour": np.sin(2 * np.pi * hour / 24.0),
            "cos_hour": np.cos(2 * np.pi * hour / 24.0),
            "day_of_week": dow,
            "is_weekend": int(dow >= 5),
            # Lags fetched directly from history
            "p_load_kw_lag1": recent_history["p_load_kw"].iloc[-1],
            "p_load_kw_lag4": recent_history["p_load_kw"].iloc[-4],
            "p_load_kw_lag96": recent_history["p_load_kw"].iloc[-96],
            "p_pv_kw_lag1": recent_history["p_pv_kw"].iloc[-1],
            "p_pv_kw_lag4": recent_history["p_pv_kw"].iloc[-4],
            "p_pv_kw_lag96": recent_history["p_pv_kw"].iloc[-96],
        }

        row_df = pd.DataFrame([feat_dict])
        
        # Inferences
        p_load_hat = float(self.load_model.predict(row_df[self.load_feature_cols])[0])
        p_pv_hat = float(self.pv_model.predict(row_df[self.pv_feature_cols])[0])

        # Physical constraints clamp (solar cannot be negative or run at night)
        p_pv_hat = max(0.0, p_pv_hat)
        if current_timestamp.hour < 5 or current_timestamp.hour > 20:
            p_pv_hat = 0.0

        p_load_hat = max(0.0, p_load_hat)

        return {
            "p_load_hat": round(p_load_hat, 2),
            "p_pv_hat": round(p_pv_hat, 2),
            "p_net_hat": round(p_load_hat - p_pv_hat, 2)
        }

    def save(self, filepath="forecast_agent.pkl"):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath="forecast_agent.pkl"):
        return joblib.load(filepath)


# =====================================================================
# Standalone Execution and Validation
# =====================================================================
if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # 1. Load the benchmark dataset generated in Step 1
    data_file = "microgrid_benchmark_7day.csv"
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        # Fallback to generating on the fly if file is not on disk
        try:
            from ..ingestion.pipeline import generate_synthetic_microgrid_dataset
        except ImportError:
            from gaaf.ingestion.pipeline import generate_synthetic_microgrid_dataset

        df = generate_synthetic_microgrid_dataset()

    print(f"Loaded benchmark dataset with {len(df)} samples.")

    # 2. Train/Test split: First 5 days for training, last 2 days for evaluation
    train_split_idx = 5 * 24 * 4  # 480 steps
    df_train = df.iloc[:train_split_idx]
    df_test = df.iloc[train_split_idx:]

    agent = ForecastAgent()
    print("Training Forecast Agent on historical telemetry...")
    agent.train(df_train)
    print("Training complete.")

    # 3. Simulate sequential sliding-window test inference
    test_features = ForecastFeatureEngineer.extract_features(df).iloc[train_split_idx:]
    y_true_load = test_features["p_load_kw"]
    y_true_pv = test_features["p_pv_kw"]

    y_pred_load = agent.load_model.predict(test_features[agent.load_feature_cols])
    y_pred_pv = np.maximum(0, agent.pv_model.predict(test_features[agent.pv_feature_cols]))

    # Zero out night-time PV predictions
    night_mask = (test_features.index.hour < 5) | (test_features.index.hour > 20)
    y_pred_pv[night_mask] = 0.0

    # 4. Compute standard accuracy metrics (defensible in paper evaluation)
    load_rmse = np.sqrt(mean_squared_error(y_true_load, y_pred_load))
    load_mae = mean_absolute_error(y_true_load, y_pred_load)
    pv_rmse = np.sqrt(mean_squared_error(y_true_pv, y_pred_pv))
    pv_mae = mean_absolute_error(y_true_pv, y_pred_pv)

    print("\n--- Forecast Agent Benchmark Accuracy (Test Horizon: 48 Hours) ---")
    print(f"Load Demand Forecast -> RMSE: {load_rmse:.2f} kW | MAE: {load_mae:.2f} kW")
    print(f"PV Solar Forecast    -> RMSE: {pv_rmse:.2f} kW | MAE: {pv_mae:.2f} kW")

    # Sample single-step inference call for the Arbitration Engine
    sample_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    sample_history = df.iloc[-100:-1]
    prediction = agent.predict(sample_ts, sample_history)
    print(f"\nNext-Interval Prediction Output: {prediction}")