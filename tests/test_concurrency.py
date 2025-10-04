import pytest

from src.simulation import (
    SimulationConfig,
    SimulationResult,
    aggregate_worker_results,
    run_simulations_concurrently,
)


def build_result(worker_id, ledger, num_games=1):
    return SimulationResult(
        worker_id=worker_id,
        seed=worker_id,
        ledger=ledger,
        q_table={},
        metrics={"num_games": num_games},
        artifact_paths=[],
        shuffle_history=[(f"sig-{worker_id}",)],
    )


def test_aggregate_worker_results_calculates_average_scores():
    ledger1 = [
        {"player_id": 0, "score": 10},
        {"player_id": 1, "score": 20},
    ]
    ledger2 = [
        {"player_id": 0, "score": 30},
        {"player_id": 1, "score": 40},
    ]
    result1 = build_result(0, ledger1)
    result2 = build_result(1, ledger2)

    aggregation = aggregate_worker_results([result1, result2])

    assert aggregation.avg_scores[0] == pytest.approx(20)
    assert aggregation.avg_scores[1] == pytest.approx(30)
    assert aggregation.worker_count == 2


@pytest.mark.parametrize("num_workers,num_games", [(1, 2), (2, 2)])
def test_run_simulations_concurrently_produces_unique_shuffles(num_workers, num_games):
    config = SimulationConfig(
        num_games=num_games,
        holes_per_game=1,
        shuffle=True,
        verbose=False,
    )

    results = run_simulations_concurrently(
        config,
        num_workers=num_workers,
        base_seed=1234,
    )

    assert len(results) == num_workers

    if num_workers > 1:
        signatures = {tuple(res.shuffle_history[0]) for res in results if res.shuffle_history}
        assert len(signatures) == num_workers

    aggregation = aggregate_worker_results(results)
    expected_rows = num_games * config.holes_per_game * 4  # 4 default players
    assert len(aggregation.ledger) == expected_rows
    assert aggregation.worker_count == num_workers
