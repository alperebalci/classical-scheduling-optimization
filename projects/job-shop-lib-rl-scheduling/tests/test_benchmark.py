import math

from jobshoplib_bridge import (
    FT06_OPTIMAL_MAKESPAN,
    build_ft06_env,
    rollout_first_ready_policy,
    run_ft06_benchmark,
)


def test_ft06_exact_reference_reaches_known_optimum():
    result = run_ft06_benchmark(max_time_seconds=5.0)
    assert result.cp_sat_makespan == FT06_OPTIMAL_MAKESPAN
    assert result.cp_sat_gap == 0
    assert result.dispatching_makespan >= FT06_OPTIMAL_MAKESPAN


def test_graph_environment_has_job_features():
    env = build_ft06_env()
    observation, _ = env.reset()
    assert "jobs" in observation
    assert observation["jobs"].shape[0] == 6
    env.close()


def test_first_ready_policy_completes_ft06():
    result = rollout_first_ready_policy()
    assert result.steps == 36
    assert result.final_makespan >= FT06_OPTIMAL_MAKESPAN
    assert math.isfinite(result.total_reward)
