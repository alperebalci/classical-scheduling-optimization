from __future__ import annotations

import random
from typing import Dict, List, Sequence

from .core import BalanceResult, Edge, TaskTimes, predecessor_map, transitive_successors, validate_instance


def _assign_by_priority(
    task_times: TaskTimes,
    precedence: Sequence[Edge],
    cycle_time: int,
    priorities: Dict[str, float],
) -> BalanceResult:
    validate_instance(task_times, precedence, cycle_time)
    pred = predecessor_map(task_times, precedence)
    unassigned = set(task_times)
    completed: set[str] = set()
    stations: List[List[str]] = []

    while unassigned:
        station: List[str] = []
        remaining = cycle_time
        while True:
            station_set = set(station)
            eligible = [
                task
                for task in unassigned
                if pred[task].issubset(completed | station_set)
                and task_times[task] <= remaining
            ]
            if not eligible:
                break
            eligible.sort(key=lambda task: (-priorities[task], -task_times[task], task))
            chosen = eligible[0]
            station.append(chosen)
            unassigned.remove(chosen)
            remaining -= task_times[chosen]

        if not station:
            raise RuntimeError("No feasible task can be assigned; check the instance.")

        stations.append(station)
        completed.update(station)

    return BalanceResult(stations=stations, cycle_time=cycle_time, task_times=task_times)


def largest_candidate_rule(
    task_times: TaskTimes,
    precedence: Sequence[Edge],
    cycle_time: int,
) -> BalanceResult:
    priorities = {task: float(time) for task, time in task_times.items()}
    return _assign_by_priority(task_times, precedence, cycle_time, priorities)


def ranked_positional_weight(
    task_times: TaskTimes,
    precedence: Sequence[Edge],
    cycle_time: int,
) -> BalanceResult:
    followers = transitive_successors(task_times, precedence)
    priorities = {
        task: float(task_times[task] + sum(task_times[follower] for follower in followers[task]))
        for task in task_times
    }
    return _assign_by_priority(task_times, precedence, cycle_time, priorities)


def comsoal(
    task_times: TaskTimes,
    precedence: Sequence[Edge],
    cycle_time: int,
    *,
    iterations: int = 500,
    seed: int = 42,
) -> BalanceResult:
    """Randomized COMSOAL construction for SALBP-1.

    Each trial repeatedly selects a precedence-feasible task that fits the
    remaining station capacity. The best trial is selected first by station
    count and then by squared idle-time imbalance.
    """
    validate_instance(task_times, precedence, cycle_time)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    pred = predecessor_map(task_times, precedence)
    rng = random.Random(seed)
    best_result: BalanceResult | None = None
    best_score: tuple[int, int, tuple[tuple[str, ...], ...]] | None = None

    for _ in range(iterations):
        unassigned = set(task_times)
        completed: set[str] = set()
        stations: List[List[str]] = []

        while unassigned:
            station: List[str] = []
            remaining = cycle_time

            while True:
                station_set = set(station)
                eligible = sorted(
                    task
                    for task in unassigned
                    if pred[task].issubset(completed | station_set)
                    and task_times[task] <= remaining
                )
                if not eligible:
                    break

                chosen = rng.choice(eligible)
                station.append(chosen)
                unassigned.remove(chosen)
                remaining -= task_times[chosen]

            if not station:
                raise RuntimeError("No feasible task can be assigned; check the instance.")

            stations.append(station)
            completed.update(station)

        result = BalanceResult(
            stations=stations,
            cycle_time=cycle_time,
            task_times=task_times,
        )
        idle_square_sum = sum(
            (cycle_time - load) ** 2 for load in result.station_loads
        )
        score = (
            result.station_count,
            idle_square_sum,
            tuple(tuple(station) for station in result.stations),
        )

        if best_score is None or score < best_score:
            best_score = score
            best_result = result

    if best_result is None:
        raise RuntimeError("COMSOAL produced no solution")
    return best_result
