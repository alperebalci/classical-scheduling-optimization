from project_management import Activity, critical_path_method, pert_mean_variance, monte_carlo_project_duration, select_project_portfolio


def test_cpm_finds_duration_and_critical_path():
    acts = {
        "A": Activity(3),
        "B": Activity(4, ("A",)),
        "C": Activity(2, ("A",)),
        "D": Activity(5, ("B", "C")),
    }
    rows, duration, critical = critical_path_method(acts)
    assert duration == 12
    assert critical == ["A", "B", "D"]
    assert rows["C"].slack == 2


def test_pert_formula():
    mean, var = pert_mean_variance(2, 5, 8)
    assert mean == 5
    assert var == 1


def test_portfolio_selection():
    chosen, value = select_project_portfolio([4, 5, 3], [7, 9, 5], budget=8)
    assert chosen == [1, 2]
    assert value == 14


def test_monte_carlo_percentiles_are_ordered():
    network = {
        "A": (2, 3, 5, ()),
        "B": (3, 4, 7, ("A",)),
        "C": (1, 2, 3, ("A",)),
    }
    result = monte_carlo_project_duration(network, n=1500, seed=4)
    assert result["p90"] >= result["p50"]
    assert result["mean"] > 0
