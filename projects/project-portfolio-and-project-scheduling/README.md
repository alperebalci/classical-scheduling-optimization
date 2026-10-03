# Project Portfolio and Project Scheduling

A Management Science companion to machine/job scheduling, focused on project networks and capital allocation.

Implemented components:

- CPM forward/backward passes;
- earliest/latest times and activity slack;
- critical-path identification with cycle/precedence validation;
- three-point PERT mean and variance formulas;
- Monte Carlo project-duration risk using three-point activity estimates;
- 0/1 project-portfolio selection under an integer capital budget.

The distinction from production scheduling is deliberate: project management deals with precedence networks, completion-risk commitments and portfolio-level capital selection rather than repeated machine sequencing alone.

Run:

```bash
python -m pip install -r requirements.txt
pytest -q
```

The portfolio solver is a transparent knapsack DP. Resource-constrained project scheduling, project crashing, endogenous precedence choices and multi-period capital constraints remain separate extensions.
