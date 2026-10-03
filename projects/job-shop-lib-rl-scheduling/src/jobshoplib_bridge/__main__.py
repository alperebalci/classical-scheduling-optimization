"""CLI for the JobShopLib FT06 benchmark."""

from __future__ import annotations

import json

from .benchmark import rollout_first_ready_policy, run_ft06_benchmark


def main() -> None:
    benchmark = run_ft06_benchmark()
    rollout = rollout_first_ready_policy()
    payload = {
        "ft06": {
            "known_optimum": benchmark.known_optimum,
            "dispatching_makespan": benchmark.dispatching_makespan,
            "dispatching_gap": benchmark.dispatching_gap,
            "cp_sat_makespan": benchmark.cp_sat_makespan,
            "cp_sat_gap": benchmark.cp_sat_gap,
        },
        "rl_environment_smoke_test": {
            "steps": rollout.steps,
            "total_reward": rollout.total_reward,
            "final_makespan": rollout.final_makespan,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
