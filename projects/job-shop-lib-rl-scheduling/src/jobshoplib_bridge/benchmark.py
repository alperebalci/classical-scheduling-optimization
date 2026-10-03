"""Framework-level FT06 benchmark using JobShopLib."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.constraint_programming import ORToolsSolver
from job_shop_lib.dispatching import (
    DispatcherObserverConfig,
    DispatchingRuleSolver,
)
from job_shop_lib.dispatching.feature_observers import (
    FeatureObserverType,
    FeatureType,
)
from job_shop_lib.graphs import build_disjunctive_graph
from job_shop_lib.reinforcement_learning import (
    ObservationSpaceKey,
    SingleJobShopGraphEnv,
)

FT06_OPTIMAL_MAKESPAN = 55


@dataclass(frozen=True)
class BenchmarkResult:
    dispatching_makespan: int
    cp_sat_makespan: int
    known_optimum: int = FT06_OPTIMAL_MAKESPAN

    @property
    def dispatching_gap(self) -> int:
        return self.dispatching_makespan - self.known_optimum

    @property
    def cp_sat_gap(self) -> int:
        return self.cp_sat_makespan - self.known_optimum


@dataclass(frozen=True)
class RolloutResult:
    steps: int
    total_reward: float
    final_makespan: int


def run_ft06_benchmark(max_time_seconds: float = 5.0) -> BenchmarkResult:
    """Compare a dispatching baseline with JobShopLib's CP-SAT wrapper."""
    if max_time_seconds <= 0:
        raise ValueError("max_time_seconds must be positive.")

    instance = load_benchmark_instance("ft06")

    dispatching = DispatchingRuleSolver("most_work_remaining").solve(instance)
    exact = ORToolsSolver(max_time_in_seconds=max_time_seconds).solve(instance)

    return BenchmarkResult(
        dispatching_makespan=int(dispatching.makespan()),
        cp_sat_makespan=int(exact.makespan()),
    )


def build_ft06_env() -> SingleJobShopGraphEnv:
    """Construct a graph-based Gymnasium environment for FT06."""
    instance = load_benchmark_instance("ft06")
    graph = build_disjunctive_graph(instance)
    feature_configs = [
        DispatcherObserverConfig(
            FeatureObserverType.IS_READY,
            kwargs={"feature_types": [FeatureType.JOBS]},
        )
    ]
    return SingleJobShopGraphEnv(
        job_shop_graph=graph,
        feature_observer_configs=feature_configs,
        render_mode=None,
    )


def rollout_first_ready_policy() -> RolloutResult:
    """Exercise the complete environment with a deterministic feasible policy."""
    env = build_ft06_env()
    observation, _ = env.reset()

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0.0

    while not (terminated or truncated):
        ready = np.flatnonzero(
            np.asarray(
                observation[ObservationSpaceKey.JOBS.value]
            ).reshape(-1)
            > 0.5
        )
        if ready.size == 0:
            raise RuntimeError("Environment exposed no ready job before termination.")

        action = (int(ready[0]), -1)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

    makespan = int(env.dispatcher.schedule.makespan())
    env.close()
    return RolloutResult(
        steps=steps,
        total_reward=total_reward,
        final_makespan=makespan,
    )
