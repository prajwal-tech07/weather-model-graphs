"""Unit tests for the scaling benchmark harness.

These focus on the repetition/median logic added for #144 (phase 2). Graph
creation and the clock are both stubbed out so the tests are deterministic and
run instantly -- we are testing the measurement bookkeeping here, not the
graph-creation code itself.
"""

import pytest

from tests.benchmarks import graph_creation_scaling as gcs


class _FakeClock:
    """A stand-in for ``time.perf_counter`` returning a scripted sequence.

    ``_measure_runtimes`` calls the clock twice per repetition (before and
    after), so a sequence of ``[0, 5, 0, 1, 0, 3]`` yields durations 5, 1, 3.
    """

    def __init__(self, values):
        self._values = list(values)
        self.calls = 0

    def __call__(self):
        value = self._values[self.calls]
        self.calls += 1
        return value


@pytest.fixture
def stub_graph_creation(monkeypatch):
    """Replace the archetype creation function with a no-op call counter."""
    calls = {"count": 0}

    def fake_create(coords=None):
        calls["count"] += 1
        return None

    monkeypatch.setattr(
        gcs.wmg.create.archetype, "create_keisler_graph", fake_create, raising=True
    )
    return calls


def _run(monkeypatch, clock_values=None, **kwargs):
    """Run a single-grid-size benchmark, optionally with a scripted clock."""
    if clock_values is not None:
        monkeypatch.setattr(gcs.time, "perf_counter", _FakeClock(clock_values))
    params = dict(min_N=4, max_N=4, num_steps=1, archetype="keisler")
    params.update(kwargs)
    return gcs.run_benchmark(**params)


def test_reports_median_not_mean_or_first(monkeypatch, stub_graph_creation):
    # durations 5, 1, 3 -> median 3 (mean would be 3.0 too, so use a skewed
    # set below as well); first-sample would be 5.
    results = _run(monkeypatch, [0, 5, 0, 1, 0, 3], repetitions=3)
    assert results[0]["runtime_s"] == 3


def test_median_ignores_a_single_outlier(monkeypatch, stub_graph_creation):
    # One pathologically slow run (100s) must not move the reported number.
    # median = 2, whereas the mean would be ~21.
    results = _run(monkeypatch, [0, 1, 0, 2, 0, 3, 0, 2, 0, 100], repetitions=5)
    assert results[0]["runtime_s"] == 2


def test_runs_create_fn_once_per_repetition(monkeypatch, stub_graph_creation):
    _run(monkeypatch, [0, 1] * 4, repetitions=4, track_memory=False)
    assert stub_graph_creation["count"] == 4


def test_memory_measured_once_regardless_of_repetitions(
    monkeypatch, stub_graph_creation
):
    # 4 timed runs + exactly 1 extra run for the memory measurement.
    _run(monkeypatch, [0, 1] * 4, repetitions=4, track_memory=True)
    assert stub_graph_creation["count"] == 5


def test_samples_and_repetitions_recorded(monkeypatch, stub_graph_creation):
    results = _run(monkeypatch, [0, 5, 0, 1, 0, 3], repetitions=3)
    record = results[0]
    assert record["repetitions"] == 3
    assert record["runtime_samples"] == [5, 1, 3]


def test_defaults_to_a_single_repetition(monkeypatch, stub_graph_creation):
    results = _run(monkeypatch, [0, 7])
    record = results[0]
    assert record["repetitions"] == 1
    assert record["runtime_s"] == 7
    assert record["runtime_samples"] == [7]
    assert stub_graph_creation["count"] == 1


def test_schema_keys_required_by_compare_are_present(monkeypatch, stub_graph_creation):
    """compare.py depends on these keys; guard against renaming them."""
    results = _run(monkeypatch, [0, 1, 0, 1], repetitions=2, track_memory=True)
    record = results[0]
    assert record["grid_points"] == 16  # N=4 -> 4*4
    assert isinstance(record["runtime_s"], float)
    assert record["peak_memory_mb"] is not None


def test_memory_is_none_when_not_tracked(monkeypatch, stub_graph_creation):
    results = _run(monkeypatch, [0, 1], track_memory=False)
    assert results[0]["peak_memory_mb"] is None


def test_rejects_non_positive_repetitions(stub_graph_creation):
    with pytest.raises(ValueError, match="repetitions must be >= 1"):
        gcs.run_benchmark(
            min_N=4, max_N=4, num_steps=1, archetype="keisler", repetitions=0
        )


def test_peak_memory_stops_tracemalloc_even_on_error(monkeypatch):
    """A failure mid-measurement must not leave tracemalloc running."""

    def boom(coords=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        gcs._measure_peak_memory(boom, xy=None)
    assert not gcs.tracemalloc.is_tracing()
