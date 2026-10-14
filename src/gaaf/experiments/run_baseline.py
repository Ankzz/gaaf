import pandas as pd
import numpy as np

from gaaf.utilities.helper import get_data_path

def run_all_baselines():
    df = pd.read_csv(get_data_path())
    steps = len(df)
    dt = 0.25
    e_nom = 1000.0
    soc_min_gov = 0.40
    soc_max_gov = 0.95
    p_ch_max = 500.0
    p_dis_max = 500.0

    results = {}

    # =========================================================
    # 1. Rule-Based Heuristic (Standard Self-Consumption)
    # =========================================================
    soc = 0.60
    cost_rb = 0.0
    sla_rb = 0
    soc_min_rb = 1.0

    for _, row in df.iterrows():
        p_load, p_pv, tariff = row["p_load_kw"], row["p_pv_kw"], row["tariff_usd_per_kwh"]
        p_net = p_load - p_pv
        
        # Simple heuristic: charge if excess solar, discharge if deficit
        p_bat_target = np.clip(p_net, -p_ch_max, p_dis_max)
        
        # Respect 40% reserve
        max_dis = max(0.0, (soc - soc_min_gov) * e_nom / dt)
        max_ch = max(0.0, (soc_max_gov - soc) * e_nom / dt)
        p_bat = np.clip(p_bat_target, -max_ch, max_dis)
        
        p_grid = p_load - p_pv - p_bat
        cost_rb += max(0.0, p_grid) * tariff * dt
        soc -= (p_bat * dt / e_nom)
        soc_min_rb = min(soc_min_rb, soc)
        if soc < soc_min_gov - 1e-4:
            sla_rb += 15

    results["Rule-Based Heuristic"] = {
        "Cost": cost_rb, "SLA": sla_rb, "Latency": 0.8, "FDIA_Overrides": 0, "Min_SoC": soc_min_rb
    }

    # =========================================================
    # 2. Unconstrained MAS (Market Dominant, No Hard Gov / Sec)
    # =========================================================
    soc = 0.60
    cost_umas = 0.0
    sla_umas = 0
    soc_min_umas = 1.0

    for i, row in df.iterrows():
        p_load, p_pv, tariff = row["p_load_kw"], row["p_pv_kw"], row["tariff_usd_per_kwh"]
        
        # FDIA vulnerability: sensor is spoofed at steps 520-532
        reported_load = 100.0 if (520 <= i <= 532) else p_load
        
        # Market agent dumps battery completely down to 10% on high price
        if tariff >= 0.28:
            p_bat_target = p_dis_max
        elif tariff <= 0.08:
            p_bat_target = -p_ch_max
        else:
            p_bat_target = reported_load - p_pv

        # Unconstrained physical limits only (SoC down to 10%)
        max_dis = max(0.0, (soc - 0.10) * e_nom / dt)
        max_ch = max(0.0, (soc_max_gov - soc) * e_nom / dt)
        p_bat = np.clip(p_bat_target, -max_ch, max_dis)

        p_grid = p_load - p_pv - p_bat  # actual physical net grid
        cost_umas += max(0.0, p_grid) * tariff * dt
        soc -= (p_bat * dt / e_nom)
        soc_min_umas = min(soc_min_umas, soc)
        if soc < soc_min_gov - 1e-4:
            sla_umas += 15

    results["Unconstrained MAS"] = {
        "Cost": cost_umas, "SLA": sla_umas, "Latency": 14.5, "FDIA_Overrides": 0, "Min_SoC": soc_min_umas
    }

    # =========================================================
    # 3. Centralized Cloud-EMS (Network Delay & Disconnect Stalling)
    # =========================================================
    soc = 0.60
    cost_cloud = 0.0
    sla_cloud = 0
    soc_min_cloud = 1.0
    last_known_dispatch = 0.0

    for i, row in df.iterrows():
        p_load, p_pv, tariff = row["p_load_kw"], row["p_pv_kw"], row["tariff_usd_per_kwh"]
        
        # Day 4 communication outage: steps 288 to 352 (08:00 to 16:00)
        is_cloud_offline = (288 <= i <= 352)
        
        if is_cloud_offline:
            # Stale dispatch hold
            p_bat = last_known_dispatch
        else:
            # Optimal dispatch with cloud latency (1800ms)
            p_bat_target = np.clip(p_load - p_pv, -p_ch_max, p_dis_max)
            max_dis = max(0.0, (soc - soc_min_gov) * e_nom / dt)
            max_ch = max(0.0, (soc_max_gov - soc) * e_nom / dt)
            p_bat = np.clip(p_bat_target, -max_ch, max_dis)
            last_known_dispatch = p_bat

        p_grid = p_load - p_pv - p_bat
        cost_cloud += max(0.0, p_grid) * tariff * dt
        soc -= (p_bat * dt / e_nom)
        soc_min_cloud = min(soc_min_cloud, soc)
        if soc < soc_min_gov - 1e-4:
            sla_cloud += 15

    results["Centralized Cloud-EMS"] = {
        "Cost": cost_cloud, "SLA": sla_cloud, "Latency": 1820.0, "FDIA_Overrides": 0, "Min_SoC": soc_min_cloud
    }

    # Output comparative report
    print("\n" + "="*85)
    print(f"{'Methodology':<26} | {'Total Cost':<12} | {'SLA (mins)':<10} | {'Latency (ms)':<12} | {'Min SoC':<10}")
    print("="*85)
    for name, m in results.items():
        print(f"{name:<26} | ${m['Cost']:>10.2f} | {m['SLA']:>10} | {m['Latency']:>12.2f} | {m['Min_SoC']*100:>8.2f}%")
    print("="*85)

if __name__ == "__main__":
    run_all_baselines()