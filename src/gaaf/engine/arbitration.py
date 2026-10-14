"""
Edge Arbitration Engine (EAE) for GAAF.
Executes deterministic conflict arbitration and enforces governance bundles.
"""
import time
import numpy as np
import cvxpy as cp

class EdgeArbitrationEngine:
    def __init__(self, policy_bundle: dict):
        self.soc_min_gov = policy_bundle.get("battery_soc_min", 0.40)
        self.soc_max_gov = policy_bundle.get("battery_soc_max", 0.95)
        self.p_ch_max = policy_bundle.get("p_ch_max", 500.0)
        self.p_dis_max = policy_bundle.get("p_dis_max", 500.0)
        self.e_nom = policy_bundle.get("e_nom_kwh", 1000.0)
        self.dt = 0.25  # 15 minutes = 0.25 hours

    def arbitrate(
        self,
        p_bat_mkt: float,
        p_bat_bat: float,
        current_soc: float,
        p_pv: float,
        p_load: float,
        tariff: float,
        sigma: float
    ) -> tuple[float, float, float]:
        """
        Solves multi-objective arbitration subject to hard governance.
        Returns: (p_bat_optimal, p_grid, latency_ms)
        """
        t_start = time.perf_counter()

        # -------------------------------------------------------------
        # Gate 1: Cyber-Physical Security Audit
        # -------------------------------------------------------------
        if sigma < 0.5:
            # Overrides market; enforces conservative self-consumption mode
            p_bat_target = min(max(p_load - p_pv, -self.p_ch_max), self.p_dis_max)
            p_bat_safe = self._clamp_to_governance(p_bat_target, current_soc)
            p_grid = p_load - p_pv - p_bat_safe
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return p_bat_safe, p_grid, latency_ms

        # -------------------------------------------------------------
        # Gate 2 & 3: Multi-Objective Convex Optimization
        # -------------------------------------------------------------
        p_bat = cp.Variable()
        p_grid = cp.Variable()

        # Hard physical & governance constraints
        constraints = [
            p_grid + p_pv + p_bat == p_load,
            p_bat >= -self.p_ch_max,
            p_bat <= self.p_dis_max,
            # Strict SoC preservation:
            current_soc - (p_bat * self.dt / self.e_nom) >= self.soc_min_gov,
            current_soc - (p_bat * self.dt / self.e_nom) <= self.soc_max_gov,
        ]

        # Multi-objective terms
        cost_market = tariff * p_grid * self.dt
        cost_degradation = 0.005 * cp.square(p_bat)
        penalty_deviation = 0.05 * cp.square(p_bat - p_bat_mkt) + 0.1 * cp.square(p_bat - p_bat_bat)

        objective = cp.Minimize(cost_market + cost_degradation + penalty_deviation)
        problem = cp.Problem(objective, constraints)

        # Solve using lightweight OSQP or ECOS
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback to deterministic heuristic if solver fails
            p_bat_val = self._clamp_to_governance(0.0, current_soc)
        else:
            p_bat_val = float(p_bat.value)

        p_bat_val = self._clamp_to_governance(p_bat_val, current_soc)
        p_grid_val = float(p_load - p_pv - p_bat_val)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return round(p_bat_val, 2), round(p_grid_val, 2), round(latency_ms, 2)

    def _clamp_to_governance(self, p_bat: float, current_soc: float) -> float:
        """Deterministic hard clamp against battery SOC limits."""
        # Max discharge allowed without breaching soc_min_gov
        max_dis = max(0.0, (current_soc - self.soc_min_gov) * self.e_nom / self.dt)
        # Max charge allowed without breaching soc_max_gov
        max_ch = max(0.0, (self.soc_max_gov - current_soc) * self.e_nom / self.dt)

        p_bat_clamped = min(max(p_bat, -min(max_ch, self.p_ch_max)), min(max_dis, self.p_dis_max))
        return p_bat_clamped