import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from gaaf.utilities.helper import get_data_path


def generate_synthetic_microgrid_dataset(
    start_date="2026-06-01 00:00:00",
    days=7,
    freq="15min",
    peak_load_kw=500.0,
    solar_capacity_kw=350.0,
    random_seed=42
):
    """
    Generates a 7-day reproducible microgrid benchmark dataset at 15-minute intervals.
    Includes:
      - Load Demand (kW) reflecting commercial dual-peak profiles.
      - Solar PV Generation (kW) following solar zenith curves with cloud intermittency.
      - Grid Tariff ($/kWh) including base rates, peak windows, and an extreme price spike event.
    """
    np.random.seed(random_seed)
    
    # Generate timestamp index
    date_range = pd.date_range(start=start_date, periods=days * 24 * 4, freq=freq)
    df = pd.DataFrame(index=date_range)
    df["timestamp"] = df.index
    
    hours = df.index.hour + df.index.minute / 60.0
    day_of_week = df.index.dayofweek  # 0: Monday, 6: Sunday
    day_index = (df.index - df.index[0]).days

    # -------------------------------------------------------------
    # 1. Commercial Load Profile Modeling (Peak ~ 500 kW)
    # -------------------------------------------------------------
    # Baseline diurnal cycle: dual peak (10:00 AM and 3:00 PM)
    base_load = 120.0  # Base baseload / overnight load in kW
    diurnal_pattern = (
        180.0 * np.exp(-((hours - 10.5) ** 2) / 12.0) +
        220.0 * np.exp(-((hours - 15.0) ** 2) / 14.0)
    )
    
    # Weekend reduction factor (30% reduction for commercial facility)
    weekend_mask = day_of_week >= 5
    weekend_factor = np.where(weekend_mask, 0.70, 1.0)
    
    # Noise and thermal fluctuation
    noise = np.random.normal(0, 15.0, len(df))
    load_kw = (base_load + diurnal_pattern) * weekend_factor + noise
    df["p_load_kw"] = np.clip(load_kw, 80.0, peak_load_kw).round(2)

    # -------------------------------------------------------------
    # 2. Solar PV Generation Modeling (Peak ~ 350 kWp)
    # -------------------------------------------------------------
    # Solar profile between 6:00 AM (hour 6) and 7:00 PM (hour 19)
    solar_window = (hours >= 6.0) & (hours <= 19.0)
    solar_shape = np.sin(np.pi * (hours - 6.0) / 13.0)
    solar_shape = np.where(solar_window, np.maximum(0, solar_shape), 0.0)
    
    # Intermittent cloud transients (Gaussian cloud covers)
    cloud_cover = np.random.beta(a=0.8, b=4.0, size=len(df))
    # Day 3 has an overcast afternoon
    day_3_mask = (day_index == 2) & (hours >= 12.0) & (hours <= 16.0)
    cloud_cover[day_3_mask] = np.random.uniform(0.6, 0.85, np.sum(day_3_mask))
    
    pv_kw = solar_capacity_kw * (solar_shape ** 1.3) * (1.0 - cloud_cover)
    df["p_pv_kw"] = np.clip(pv_kw, 0.0, solar_capacity_kw).round(2)

    # -------------------------------------------------------------
    # 3. Dynamic Tariff Modeling ($/kWh)
    # -------------------------------------------------------------
    # Off-peak: $0.08/kWh, Shoulder: $0.14/kWh, On-peak (14:00-18:00): $0.28/kWh
    tariff = np.full(len(df), 0.08)
    shoulder_mask = ((hours >= 8.0) & (hours < 14.0)) | ((hours >= 18.0) & (hours < 22.0))
    on_peak_mask = (hours >= 14.0) & (hours < 18.0)
    
    tariff[shoulder_mask] = 0.14
    tariff[on_peak_mask] = 0.28
    
    # EVENT 1: Price Volatility Spike on Day 2 between 14:00 and 17:00 (5.2x jump)
    spike_mask = (day_index == 1) & (hours >= 14.0) & (hours <= 17.0)
    tariff[spike_mask] = 0.28 * 5.2  # ~$1.456 / kWh
    
    df["tariff_usd_per_kwh"] = np.round(tariff, 4)
    
    # -------------------------------------------------------------
    # 4. Net Demand (P_net = P_load - P_pv)
    # -------------------------------------------------------------
    df["p_net_kw"] = (df["p_load_kw"] - df["p_pv_kw"]).round(2)

    return df


def preprocess_real_datasets(load_csv_path, tariff_csv_path=None, target_peak_kw=500.0):
    """
    Utility helper to load, resample, and scale raw PJM or CAISO CSV data.
    Expected load_csv format: columns ['datetime', 'load']
    """
    df_load = pd.read_csv(load_csv_path, parse_dates=["datetime"], index_col="datetime")
    
    # Resample to 15-minute intervals using linear interpolation
    df_resampled = df_load.resample("15min").interpolate(method="time")
    
    # Scale to 500 kW commercial microgrid capacity
    max_val = df_resampled["load"].max()
    df_resampled["p_load_kw"] = (df_resampled["load"] / max_val) * target_peak_kw
    
    return df_resampled


if __name__ == "__main__":
    # Generate the 7-day dataset
    benchmark_df = generate_synthetic_microgrid_dataset()
    
    # Save to CSV for Steps 2, 3, and 4
    output_path = get_data_path()
    benchmark_df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully created and exported to '{output_path}'.")
    print(f"Total time steps: {len(benchmark_df)} (Interval: 15 minutes, Days: 7)")
    print("\nSample Data (First 5 Rows):")
    print(benchmark_df[["timestamp", "p_load_kw", "p_pv_kw", "tariff_usd_per_kwh", "p_net_kw"]].head())
    
    print("\nSummary Statistics:")
    print(benchmark_df[["p_load_kw", "p_pv_kw", "tariff_usd_per_kwh"]].describe())