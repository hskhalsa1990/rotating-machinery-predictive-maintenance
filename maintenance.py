"""Maintenance decision logic, cost model and a small reinforcement-learning agent.

All costs (INR) are ILLUSTRATIVE ASSUMPTIONS, not real plant data.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostParams:
    planned_repair: float = 35_000.0
    planned_downtime_h: float = 8.0
    unplanned_repair: float = 250_000.0
    unplanned_downtime_h: float = 48.0
    downtime_cost_per_h: float = 6_000.0

    @property
    def planned_total(self) -> float:
        return self.planned_repair + self.planned_downtime_h * self.downtime_cost_per_h

    @property
    def unplanned_total(self) -> float:
        return self.unplanned_repair + self.unplanned_downtime_h * self.downtime_cost_per_h


def failure_probability(rul_samples, horizon: float) -> float:
    return float(np.mean(np.asarray(rul_samples) <= horizon))


def expected_costs(p_fail: float, rul_median: float, cp: CostParams = CostParams(), interval: float = 180.0) -> dict:
    """Repair now (loses unused life) vs wait (risk of an unplanned failure)."""
    wasted = cp.planned_total * min(max(rul_median, 0.0), interval) / interval
    return {"repair_now": cp.planned_total + wasted,
            "wait": p_fail * cp.unplanned_total + (1 - p_fail) * cp.planned_total}


def plan_action(zone_letter: str, rul: dict, lead_time: float = 7, window: float = 14, p_urgent: float = 0.2) -> str:
    """Decision rule (same as the flowchart in the README)."""
    if zone_letter == "D":
        return "STOP_AND_INSPECT_NOW"
    if failure_probability(rul["samples"], lead_time) >= p_urgent:
        return "URGENT_REPAIR"
    if rul["lo"] <= lead_time + window:
        return "PLAN_NEXT_WINDOW"
    return "MONITOR"


# ---------------- Reinforcement learning: keep running vs repair ----------------
N_STATES = 6                      # 0 = like new ... 4 = badly worn, 5 = failed
P_DEGRADE = [0.05, 0.10, 0.18, 0.30, 0.45]
REVENUE, REVENUE_LOSS_PER_STATE = 100.0, 10.0
REPAIR_COST, FAIL_COST = 60.0, 400.0


def step(s: int, a: int, rng):
    """a: 0 = run, 1 = repair. Returns (next_state, reward)."""
    if s == N_STATES - 1:
        return 0, -FAIL_COST
    if a == 1:
        return 0, -REPAIR_COST
    r = REVENUE - REVENUE_LOSS_PER_STATE * s
    return (s + 1 if rng.random() < P_DEGRADE[s] else s), r


def value_iteration(gamma: float = 0.95, iters: int = 500):
    """Exact solution of the same MDP (cross-check for Q-learning)."""
    Q = np.zeros((N_STATES, 2))
    for _ in range(iters):
        V = Q.max(1)
        for s in range(N_STATES):
            if s == N_STATES - 1:
                Q[s, :] = -FAIL_COST + gamma * V[0]
                continue
            Q[s, 1] = -REPAIR_COST + gamma * V[0]
            p = P_DEGRADE[s]
            Q[s, 0] = REVENUE - REVENUE_LOSS_PER_STATE * s + gamma * (p * V[s + 1] + (1 - p) * V[s])
    return Q


def q_learning(episodes: int = 3000, T: int = 50, gamma: float = 0.95, alpha: float = 0.1,
               eps_start: float = 1.0, eps_end: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, 2))
    returns = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / (0.7 * episodes)))
        s, tot = 0, 0.0
        for _ in range(T):
            a = int(rng.integers(2)) if rng.random() < eps else int(Q[s].argmax())
            s2, r = step(s, a, rng)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            s, tot = s2, tot + r
        returns.append(tot)
    return Q, np.array(returns)


def evaluate_policy(policy, episodes: int = 300, T: int = 200, seed: int = 123):
    """policy: array/list mapping state -> action. Returns (mean reward per step, failures per 100 steps)."""
    rng = np.random.default_rng(seed)
    tot, fails = 0.0, 0
    for _ in range(episodes):
        s = 0
        for _ in range(T):
            if s == N_STATES - 1:
                fails += 1
            s, r = step(s, int(policy[s]), rng)
            tot += r
    return tot / (episodes * T), 100.0 * fails / (episodes * T)
