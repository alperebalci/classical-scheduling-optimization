from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Iterable
import numpy as np


@dataclass(frozen=True)
class Activity:
    duration: float
    predecessors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CPMRow:
    earliest_start: float
    earliest_finish: float
    latest_start: float
    latest_finish: float
    slack: float


def _topological_order(activities: Mapping[str, Activity]) -> list[str]:
    remaining = {k: set(v.predecessors) for k, v in activities.items()}
    unknown = set().union(*remaining.values()) - set(activities) if remaining else set()
    if unknown:
        raise ValueError(f"Unknown predecessors: {sorted(unknown)}")
    order = []
    while remaining:
        ready = sorted(k for k, deps in remaining.items() if deps.issubset(order))
        if not ready:
            raise ValueError("Activity graph contains a cycle")
        for k in ready:
            order.append(k); remaining.pop(k)
    return order


def critical_path_method(activities: Mapping[str, Activity]) -> tuple[dict[str, CPMRow], float, list[str]]:
    order = _topological_order(activities)
    es, ef = {}, {}
    for a in order:
        es[a] = max((ef[p] for p in activities[a].predecessors), default=0.0)
        ef[a] = es[a] + activities[a].duration
    project_duration = max(ef.values(), default=0.0)
    successors = {a: [] for a in activities}
    for a, spec in activities.items():
        for p in spec.predecessors:
            successors[p].append(a)
    ls, lf = {}, {}
    for a in reversed(order):
        lf[a] = min((ls[s] for s in successors[a]), default=project_duration)
        ls[a] = lf[a] - activities[a].duration
    rows = {a: CPMRow(es[a], ef[a], ls[a], lf[a], ls[a] - es[a]) for a in order}
    critical = [a for a in order if abs(rows[a].slack) < 1e-9]
    return rows, float(project_duration), critical


def pert_mean_variance(optimistic: float, most_likely: float, pessimistic: float) -> tuple[float, float]:
    if not optimistic <= most_likely <= pessimistic:
        raise ValueError("Require optimistic <= most_likely <= pessimistic")
    mean = (optimistic + 4 * most_likely + pessimistic) / 6.0
    variance = ((pessimistic - optimistic) / 6.0) ** 2
    return float(mean), float(variance)


def monte_carlo_project_duration(network: Mapping[str, tuple[float, float, float, tuple[str, ...]]], n=10000, seed=2026):
    rng = np.random.default_rng(seed)
    durations = np.empty(n)
    keys = list(network)
    samples = {k: rng.triangular(network[k][0], network[k][1], network[k][2], size=n) for k in keys}
    for i in range(n):
        acts = {k: Activity(float(samples[k][i]), network[k][3]) for k in keys}
        _, durations[i], _ = critical_path_method(acts)
    return {
        "mean": float(durations.mean()),
        "p50": float(np.quantile(durations, 0.50)),
        "p90": float(np.quantile(durations, 0.90)),
        "samples": durations,
    }


def select_project_portfolio(costs: Iterable[int], values: Iterable[float], budget: int) -> tuple[list[int], float]:
    costs = list(map(int, costs)); values = list(map(float, values))
    if len(costs) != len(values) or budget < 0 or any(c < 0 for c in costs):
        raise ValueError("Invalid costs/values/budget")
    n = len(costs)
    dp = np.zeros((n + 1, budget + 1))
    take = np.zeros((n + 1, budget + 1), dtype=bool)
    for i in range(1, n + 1):
        c, v = costs[i-1], values[i-1]
        for b in range(budget + 1):
            dp[i, b] = dp[i-1, b]
            if c <= b and dp[i-1, b-c] + v > dp[i, b] + 1e-12:
                dp[i, b] = dp[i-1, b-c] + v
                take[i, b] = True
    chosen = []
    b = budget
    for i in range(n, 0, -1):
        if take[i, b]:
            chosen.append(i-1); b -= costs[i-1]
    chosen.reverse()
    return chosen, float(dp[n, budget])
