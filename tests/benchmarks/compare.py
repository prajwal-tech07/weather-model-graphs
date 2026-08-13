"""Compare two scaling-benchmark JSON outputs and render a regression report.

This consumes the JSON produced by ``graph_creation_scaling.py --output-json``
(added in #140), i.e. a list of records of the form::

    [{"grid_points": int, "runtime_s": float, "peak_memory_mb": float | null}, ...]

Given a *baseline* run (typically ``main``) and a *contender* run (the PR), it
matches records by ``grid_points`` and computes the relative change in both
runtime and peak memory usage (when both runs recorded memory, i.e. were run
with ``--track-memory``), rendering a Markdown table suitable for posting as a
sticky pull-request comment.

The design follows the discussion in
https://github.com/mllam/weather-model-graphs/issues/144: we compare relative
(%) change rather than absolute seconds, because the two runs are executed
back-to-back on the *same* CI runner and only the library under test differs, so
the per-runner noise largely cancels.

Usage::

    python -m tests.benchmarks.compare main.json pr.json
    python -m tests.benchmarks.compare main.json pr.json \\
        --threshold-pct 0.1 --output comment.md --fail-on-regression

The script only depends on the standard library so it can run in a minimal CI
step without installing the package.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

# Hidden marker used by the CI workflow to find-and-update a single sticky
# comment instead of posting a new comment on every run.
STICKY_MARKER = "<!-- benchmark-regression-check -->"


def load_results(path: str) -> Dict[int, dict]:
    """Load a benchmark JSON file and index the records by ``grid_points``.

    Raises ``ValueError`` with an actionable message if the file is empty or
    does not match the expected schema, so CI failures are easy to diagnose.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"{path!r} does not contain a non-empty list of benchmark records"
        )

    indexed: Dict[int, dict] = {}
    for record in data:
        if "grid_points" not in record or "runtime_s" not in record:
            raise ValueError(
                f"{path!r} contains a record missing required keys "
                f"'grid_points'/'runtime_s': {record!r}"
            )
        indexed[int(record["grid_points"])] = record
    return indexed


def _pct_change(baseline: float, contender: float) -> Optional[float]:
    """Return the percentage change from ``baseline`` to ``contender``.

    Returns ``None`` when the baseline is zero (or negative), since a relative
    change is undefined there and we would rather skip the row than divide by
    zero.
    """
    if baseline <= 0:
        return None
    return (contender - baseline) / baseline * 100.0


class Row:
    """A single grid-size comparison line in the report.

    Memory fields are ``None`` when either run didn't record
    ``peak_memory_mb`` (i.e. wasn't run with ``--track-memory``) for this grid
    size; the memory columns are omitted from the report entirely when no row
    has memory data at all.
    """

    def __init__(
        self,
        grid_points: int,
        baseline_s: float,
        contender_s: float,
        delta_pct: Optional[float],
        is_regression: bool,
        baseline_mem_mb: Optional[float] = None,
        contender_mem_mb: Optional[float] = None,
        mem_delta_pct: Optional[float] = None,
        mem_is_regression: bool = False,
    ):
        self.grid_points = grid_points
        self.baseline_s = baseline_s
        self.contender_s = contender_s
        self.delta_pct = delta_pct
        self.is_regression = is_regression
        self.baseline_mem_mb = baseline_mem_mb
        self.contender_mem_mb = contender_mem_mb
        self.mem_delta_pct = mem_delta_pct
        self.mem_is_regression = mem_is_regression

    @property
    def has_memory(self) -> bool:
        return self.baseline_mem_mb is not None and self.contender_mem_mb is not None

    @property
    def any_regression(self) -> bool:
        return self.is_regression or self.mem_is_regression


def compare(
    baseline: Dict[int, dict],
    contender: Dict[int, dict],
    threshold_pct: float,
) -> List[Row]:
    """Build the per-grid-size comparison rows for runtime and peak memory.

    Only ``grid_points`` present in *both* runs are compared; unmatched sizes
    are skipped (they show up in the PR diff of the benchmark itself, so there is
    nothing to compare against). Rows are returned sorted by ``grid_points``.

    The memory delta for a row is only computed when *both* the baseline and
    contender records have a non-null ``peak_memory_mb`` (i.e. both runs used
    ``--track-memory``); otherwise the row's memory fields stay ``None``.
    """
    common = sorted(set(baseline) & set(contender))
    rows: List[Row] = []
    for gp in common:
        b_record = baseline[gp]
        c_record = contender[gp]

        b = float(b_record["runtime_s"])
        c = float(c_record["runtime_s"])
        delta = _pct_change(b, c)
        is_regression = delta is not None and delta > threshold_pct

        b_mem = b_record.get("peak_memory_mb")
        c_mem = c_record.get("peak_memory_mb")
        mem_delta = None
        mem_is_regression = False
        if b_mem is not None and c_mem is not None:
            b_mem = float(b_mem)
            c_mem = float(c_mem)
            mem_delta = _pct_change(b_mem, c_mem)
            mem_is_regression = mem_delta is not None and mem_delta > threshold_pct

        rows.append(
            Row(
                gp,
                b,
                c,
                delta,
                is_regression,
                b_mem,
                c_mem,
                mem_delta,
                mem_is_regression,
            )
        )
    return rows


def _fmt_seconds(value: float) -> str:
    """Format a runtime in seconds with millisecond-level readability."""
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.3f}s"


def _fmt_delta(delta: Optional[float], is_regression: bool) -> str:
    if delta is None:
        return "n/a"
    icon = "⚠️" if is_regression else "✅"
    return f"{delta:+.1f}% {icon}"


def _fmt_mb(value: float) -> str:
    """Format a peak-memory reading in MB."""
    return f"{value:.1f}MB"


def render_markdown(
    rows: List[Row],
    threshold_pct: float,
    baseline_label: str,
    contender_label: str,
    unmatched: Optional[List[int]] = None,
) -> str:
    """Render the comparison as a Markdown block, prefixed with the sticky marker."""
    lines = [STICKY_MARKER, "## ⏱️ Graph-creation benchmark: regression check", ""]

    if not rows:
        lines.append(
            "No overlapping grid sizes to compare between "
            f"`{baseline_label}` and `{contender_label}`."
        )
        return "\n".join(lines) + "\n"

    has_memory = any(r.has_memory for r in rows)
    metric_label = "runtime/memory" if has_memory else "runtime"

    regressions = [r for r in rows if r.any_regression]
    if regressions:
        lines.append(
            f"⚠️ **{len(regressions)} of {len(rows)}** grid sizes exceed the "
            f"**+{threshold_pct:g}%** {metric_label} threshold."
        )
    else:
        lines.append(
            f"✅ No {metric_label} regression above **+{threshold_pct:g}%** "
            f"across {len(rows)} grid sizes."
        )
    lines.append("")

    header = ["grid points", baseline_label, contender_label, "Δ runtime"]
    if has_memory:
        header += [
            f"{baseline_label} peak mem",
            f"{contender_label} peak mem",
            "Δ memory",
        ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")

    for r in rows:
        row_cells = [
            f"{r.grid_points:,}",
            _fmt_seconds(r.baseline_s),
            _fmt_seconds(r.contender_s),
            _fmt_delta(r.delta_pct, r.is_regression),
        ]
        if has_memory:
            if r.has_memory:
                row_cells += [
                    _fmt_mb(r.baseline_mem_mb),
                    _fmt_mb(r.contender_mem_mb),
                    _fmt_delta(r.mem_delta_pct, r.mem_is_regression),
                ]
            else:
                row_cells += ["n/a", "n/a", "n/a"]
        lines.append("| " + " | ".join(row_cells) + " |")

    if unmatched:
        pretty = ", ".join(f"{gp:,}" for gp in unmatched)
        lines.append("")
        lines.append(
            f"_Note: {len(unmatched)} grid size(s) not present in both runs were "
            f"skipped: {pretty}._"
        )

    lines.append("")
    lines.append(
        "_Runs execute back-to-back on the same runner; only the library under "
        "test differs, so relative change is compared rather than absolute time._"
    )
    return "\n".join(lines) + "\n"


def build_report(
    baseline_path: str,
    contender_path: str,
    threshold_pct: float,
    baseline_label: str,
    contender_label: str,
):
    """Load both files, compute rows, and return ``(markdown, rows)``."""
    baseline = load_results(baseline_path)
    contender = load_results(contender_path)
    rows = compare(baseline, contender, threshold_pct)
    unmatched = sorted(set(baseline) ^ set(contender))
    markdown = render_markdown(
        rows, threshold_pct, baseline_label, contender_label, unmatched
    )
    return markdown, rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two scaling-benchmark JSON outputs (baseline vs PR)."
    )
    parser.add_argument("baseline", help="Baseline JSON (e.g. main.json).")
    parser.add_argument("contender", help="Contender JSON (e.g. pr.json).")
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.1,
        help="Flag a grid size when its runtime or (if tracked) peak memory "
        "increases by more than this percentage. Start low and raise it once "
        "the runner's noise floor is known (default: 0.1).",
    )
    parser.add_argument(
        "--baseline-label", default="main", help="Column label for the baseline."
    )
    parser.add_argument(
        "--contender-label", default="PR", help="Column label for the contender."
    )
    parser.add_argument(
        "--output",
        help="Also write the Markdown report to this file (for the CI comment).",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any grid size regresses (off by default so the "
        "check is informational).",
    )
    return parser.parse_args(argv)


def _print_utf8(text: str) -> None:
    """Print ``text`` as UTF-8 regardless of the console's default encoding.

    The report contains emoji (✅/⚠️); on a Windows console (cp1252) a plain
    ``print`` raises ``UnicodeEncodeError``. CI runners are UTF-8, but we keep
    the tool robust for local use.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass
    try:
        print(text)
    except UnicodeEncodeError:
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            buffer.write(text.encode("utf-8") + b"\n")
        else:  # pragma: no cover - extremely unusual stdout replacement
            print(text.encode("utf-8", "backslashreplace").decode("ascii"))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    markdown, rows = build_report(
        args.baseline,
        args.contender,
        args.threshold_pct,
        args.baseline_label,
        args.contender_label,
    )

    _print_utf8(markdown)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)

    if args.fail_on_regression and any(r.any_regression for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
