"""
Domain and Security Agents for GAAF.
"""
import numpy as np

class MarketAgent:
    """
    Market / Tariff Intelligence Agent (A_mkt).
    Formulates aggressive economic arbitrage proposals based on dynamic tariffs.
    """
    def __init__(self, p_ch_max: float = 500.0, p_dis_max: float = 500.0):
        self.p_ch_max = p_ch_max
        self.p_dis_max = p_dis_max

    def propose_dispatch(self, tariff_usd: float, p_net_hat: float) -> float:
        """
        Returns battery power proposal (kW).
        Positive = Discharge (sell/offset grid), Negative = Charge (buy/store).
        """
        # Dynamic pricing thresholds ($/kWh)
        if tariff_usd >= 0.28:
            # Peak or Spike: Aggressively discharge to maximum capacity
            return self.p_dis_max
        elif tariff_usd <= 0.08:
            # Low tariff / Off-peak: Maximize charging from grid
            return -self.p_ch_max
        else:
            # Shoulder hours: Cover net load locally if possible
            return float(np.clip(p_net_hat, -self.p_ch_max, self.p_dis_max))


class StorageAgent:
    """
    Battery Energy Storage System Health Agent (A_bat).
    Prioritizes cell longevity, cycle degradation mitigation, and thermal health.
    """
    def __init__(self, p_ch_max: float = 500.0, p_dis_max: float = 500.0):
        self.p_ch_max = p_ch_max
        self.p_dis_max = p_dis_max

    def propose_dispatch(self, current_soc: float) -> float:
        """
        Recommends conservative power to keep SoC in optimal resting zone [0.45, 0.75].
        """
        if current_soc > 0.75:
            # Relieve high voltage stress with mild discharge
            return 100.0
        elif current_soc < 0.45:
            # Prevent deep discharge stress with mild charge
            return -100.0
        else:
            # Idle / zero cyclic wear
            return 0.0


class SecurityAgent:
    """
    Cyber-Physical Security & Telemetry Verification Agent (A_sec).
    Monitors telemetry residuals to detect False Data Injection Attacks (FDIAs).
    """
    def __init__(self, residual_threshold: float = 80.0):
        self.residual_threshold = residual_threshold

    def audit_telemetry(self, reported_load: float, physics_load_est: float) -> float:
        """
        Computes data authenticity score sigma in [0, 1].
        Uses a sigmoid decay over measurement residuals.
        """
        residual = abs(reported_load - physics_load_est)
        # Sigmoid penalty: drops near 0 if residual >> threshold
        sigma = 1.0 / (1.0 + np.exp((residual - self.residual_threshold) / 15.0))
        return float(np.round(sigma, 4))