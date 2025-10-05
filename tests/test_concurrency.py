import pytest
from pathlib import Path

from src import simulation
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

def test_run_simulations_concurrently_falls_back_to_sequential(monkeypatch):
    config = SimulationConfig(num_games=2, holes_per_game=1)

    def fake_get_context(method):
        raise PermissionError('queue creation blocked')

    recorded_calls = []

    def stub_run_simulation(*, config, seed, worker_id, game_offset, output_dir):
        result = SimulationResult(
            worker_id=worker_id,
            seed=seed,
            ledger=[{'worker': worker_id, 'offset': game_offset}],
            q_table={},
            metrics={'num_games': config.num_games},
            artifact_paths=[],
            shuffle_history=[],
        )
        recorded_calls.append(result)
        return result

    monkeypatch.setattr(simulation.mp, 'get_context', fake_get_context)
    monkeypatch.setattr(simulation, 'run_simulation', stub_run_simulation)

    results = run_simulations_concurrently(config, num_workers=2, base_seed=42)

    assert len(results) == 2
    assert [res.worker_id for res in results] == [0, 1]
    assert recorded_calls == results

def test_run_simulation_writes_artifacts(tmp_path, monkeypatch):
    config = SimulationConfig(num_games=1, holes_per_game=1)

    def stub_play_game(golf, game_num, hole, q_table, model, rank_cutoff, **kwargs):
        return ([{
            'worker_id': 3,
            'score': 5,
            'hole': hole,
            'game': game_num,
            'reward': 0,
        }], (('sig', 'value'),))

    monkeypatch.setattr(simulation, 'torch', None)
    monkeypatch.setattr(simulation, 'play_game', stub_play_game)

    result = simulation.run_simulation(
        config,
        seed=7,
        worker_id=3,
        game_offset=0,
        output_dir=str(tmp_path),
    )

    ledger_path = tmp_path / 'ledger_worker_3.csv'
    q_table_path = tmp_path / 'q_table_worker_3.json'

    assert ledger_path.exists()
    assert q_table_path.exists()
    artifact_set = {Path(p) for p in result.artifact_paths}
    assert ledger_path in artifact_set
    assert q_table_path in artifact_set

    contents = ledger_path.read_text().strip().splitlines()
    assert len(contents) == 2  # header + one record
    assert 'score' in contents[0]
    assert '5' in contents[1]

def test_worker_entry_places_result_on_queue(monkeypatch):
    expected = SimulationResult(
        worker_id=2,
        seed=99,
        ledger=[],
        q_table={},
        metrics={},
        artifact_paths=[],
        shuffle_history=[],
    )

    def stub_run_simulation(*, config, seed, worker_id, game_offset, output_dir):
        assert seed == 11
        assert worker_id == 2
        assert game_offset == 5
        assert output_dir == 'out'
        return expected

    class DummyQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    queue = DummyQueue()
    monkeypatch.setattr(simulation, 'run_simulation', stub_run_simulation)

    simulation._worker_entry(
        worker_id=2,
        config=SimulationConfig(),
        seed=11,
        game_offset=5,
        output_dir='out',
        queue=queue,
    )

    assert queue.items == [expected]
