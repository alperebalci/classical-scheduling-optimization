# JobShopLib RL-Ready Scheduling

A compact framework-oriented Job Shop Scheduling benchmark built on **job-shop-lib 1.7.1**.

## Purpose

This project is deliberately different from the repository root's native OR-Tools CP-SAT model. It asks what a dedicated scheduling framework gives us out of the box:

- canonical benchmark loading (`ft06`);
- dispatching-rule baselines;
- a CP-SAT reference solver behind one scheduling API;
- disjunctive-graph construction;
- a Gymnasium-compatible graph environment suitable for later RL or imitation-learning work.

It intentionally stops before adding another PPO/DQN implementation. The value is the reusable scheduling environment and framework-level benchmark, not another policy-gradient copy.

## Benchmark

`run_ft06_benchmark()` solves Fisher-Thompson FT06 with:

1. Most Work Remaining dispatching;
2. JobShopLib's OR-Tools solver.

FT06 has a known optimal makespan of 55, so the exact-solver result is checked against a known reference rather than treated as correct merely because it is feasible.

## RL environment smoke test

`build_ft06_env()` constructs `SingleJobShopGraphEnv` using:

- a disjunctive graph;
- an `IS_READY` job-level feature;
- the library's default makespan reward;
- no rendering side effects.

`rollout_first_ready_policy()` then exercises the complete Gymnasium step/reset loop with a deterministic first-ready policy. It is a mechanics/feasibility smoke test, not a learned agent.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
jobshoplib-benchmark
```

or

```bash
python -m jobshoplib_bridge
```

## Test

```bash
pytest
```

## Claims boundary

JobShopLib is used as an explicit framework dependency. This project does not claim that its dispatching baseline is competitive with specialized metaheuristics or learned policies, and it does not duplicate the repository root's custom CP-SAT formulation.
