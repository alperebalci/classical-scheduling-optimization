from pathlib import Path

from alb_optimizer import (
    comsoal,
    largest_candidate_rule,
    ranked_positional_weight,
    solve_salbp1,
    validate_solution,
)
from alb_optimizer.io import load_precedence, load_tasks

ROOT = Path(__file__).resolve().parents[1]
TASKS = load_tasks(ROOT / "data" / "tasks.csv")
PRECEDENCE = load_precedence(ROOT / "data" / "precedence.csv")
CYCLE_TIME = 15


def test_heuristics_are_feasible():
    heuristic_results = [
        largest_candidate_rule(TASKS, PRECEDENCE, CYCLE_TIME),
        ranked_positional_weight(TASKS, PRECEDENCE, CYCLE_TIME),
        comsoal(TASKS, PRECEDENCE, CYCLE_TIME, iterations=100, seed=7),
    ]
    for result in heuristic_results:
        validate_solution(result, PRECEDENCE)
        assert result.station_count >= result.theoretical_minimum


def test_comsoal_is_reproducible_and_reaches_lower_bound_on_fixture():
    first = comsoal(TASKS, PRECEDENCE, CYCLE_TIME, iterations=100, seed=42)
    second = comsoal(TASKS, PRECEDENCE, CYCLE_TIME, iterations=100, seed=42)

    validate_solution(first, PRECEDENCE)
    assert first.stations == second.stations
    assert first.station_count == first.theoretical_minimum == 7


def test_comsoal_rejects_nonpositive_iteration_budget():
    try:
        comsoal(TASKS, PRECEDENCE, CYCLE_TIME, iterations=0)
    except ValueError as exc:
        assert "iterations" in str(exc)
    else:
        raise AssertionError("COMSOAL must reject a nonpositive iteration budget")


def test_exact_solver_reaches_proven_lower_bound():
    result = solve_salbp1(TASKS, PRECEDENCE, CYCLE_TIME)
    validate_solution(result, PRECEDENCE)
    assert result.theoretical_minimum == 7
    assert result.station_count == 7
