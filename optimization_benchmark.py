"""Reproducible, equal-budget BO/GA comparisons on opfunu functions.

Run with the project Python, or call run_suite() from 优化算法.ipynb.
Each independent case/seed is cached so interrupted experiments can resume.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from threadpoolctl import threadpool_limits

BUDGET = 120
INITIAL_POINTS = 20
SEEDS = (0, 1, 42)


@dataclass(frozen=True)
class BenchmarkCase:
    family: str
    module: str
    classname: str
    ndim: int
    category: str
    description: str
    fixed_dimension: bool = False

    @property
    def name(self) -> str:
        return f"{self.family}_{self.ndim}D"

    @property
    def dimension_band(self) -> str:
        return "low (2-5)" if self.ndim <= 5 else "medium (10-20)" if self.ndim <= 20 else "high (30-50)"

    def create(self) -> Any:
        cls = getattr(importlib.import_module(self.module), self.classname)
        with contextlib.redirect_stdout(io.StringIO()):
            # Passing the actual fixed dimension avoids opfunu 1.0.1 leaving
            # Corana.ndim at the base default (2) while its bounds have 4 rows.
            return cls(ndim=self.ndim)


def make_cases() -> list[BenchmarkCase]:
    cases = []
    classic = [
        ("ChungReynolds", "c_func", [2, 3, 5, 10, 20, 30], "smooth_coupled", "Quartic bowl; coupled through squared sum"),
        ("Cigar", "c_func", [2, 5, 10, 20, 30, 50], "ill_conditioned", "Separable quadratic; condition ratio 1e6"),
        ("Brown", "b_func", [2, 5, 10], "smooth_coupled", "Neighboring variables coupled in exponents"),
        ("DixonPrice", "d_func", [2, 5, 10, 20], "valley", "Nonlinear coupled valley"),
        ("Ackley01", "a_func", [2, 3, 5, 10, 20, 30, 50], "multimodal", "Many local minima and a central basin"),
        ("Griewank", "g_func", [2, 5, 10, 20, 30, 50], "multimodal", "Many local minima; multiplicative coupling"),
        ("Levy03", "l_func", [2, 5, 10, 20], "multimodal", "Oscillatory nonlinear landscape"),
    ]
    for family, module, dimensions, category, description in classic:
        for dimension in dimensions:
            cases.append(BenchmarkCase(family, f"opfunu.name_based.{module}", family, dimension, category, description))
    cases.extend([
        BenchmarkCase("Beale", "opfunu.name_based.b_func", "Beale", 2, "valley", "Fixed 2D nonlinear valley", True),
        BenchmarkCase("Branin01", "opfunu.name_based.b_func", "Branin01", 2, "multiple_global_minima", "Fixed 2D; three global minima", True),
    ])
    cec = [
        ("CEC_F1", "F12014", [10, 30], "ill_conditioned_rotated", "Shifted/rotated high-conditioned elliptic"),
        ("CEC_F2", "F22014", [10], "ill_conditioned_rotated", "Shifted/rotated bent cigar"),
        ("CEC_F3", "F32014", [10, 30], "ill_conditioned_rotated", "Shifted/rotated discus"),
        ("CEC_F4", "F42014", [10, 30], "valley_rotated", "Shifted/rotated Rosenbrock"),
        ("CEC_F5", "F52014", [10], "multimodal_rotated", "Shifted/rotated Ackley"),
        ("CEC_F7", "F72014", [10], "multimodal_rotated", "Shifted/rotated Griewank"),
        ("CEC_F9", "F92014", [10, 20, 30, 50], "multimodal_rotated", "Shifted/rotated Rastrigin"),
        ("CEC_F6", "F62014", [10], "multimodal_rotated", "Shifted/rotated Weierstrass; rugged oscillations"),
        ("CEC_F19", "F192014", [10, 30], "hybrid", "Hybrid of Griewank, Weierstrass, Rosenbrock and Scaffer"),
        ("CEC_F23", "F232014", [10, 30], "composition", "Weighted composition of transformed functions"),
    ]
    for family, classname, dimensions, category, description in cec:
        for dimension in dimensions:
            cases.append(BenchmarkCase(family, "opfunu.cec_based.cec2014", classname, dimension, category, description))
    assert len(cases) == 56 and len({case.family for case in cases}) == 19
    cases.extend(make_discontinuous_cases())
    return cases


def make_discontinuous_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase("Corana", "opfunu.name_based.c_func", "Corana", 4, "discontinuous", "Piecewise, rounded landscape; fixed 4D", True),
        *[BenchmarkCase("NeedleEye", "opfunu.name_based.n_func", "NeedleEye", dimension, "discontinuous", "Very narrow optimum region and discontinuous penalties") for dimension in [2, 5, 10]],
    ]


def configuration() -> dict:
    return {
        "schema": 1,
        "budget": BUDGET,
        "initial_points": INITIAL_POINTS,
        "seeds": list(SEEDS),
        "bounds": "opfunu full default bounds",
        "bo": {"acquisition": "UCB", "kappa": 2.576, "input_scaling": "unit cube", "gp": "library defaults", "acquisition_optimization": "library defaults"},
        "ga": {"pop_size": INITIAL_POINTS, "n_offsprings": INITIAL_POINTS, "sampling": "shared BO initial points", "crossover": "default SBX", "mutation": "default PM", "eliminate_duplicates": True},
        "random": "same 20 initial points plus 100 independent uniform points",
        "versions": {name: version(name) for name in ["bayesian-optimization", "pymoo", "opfunu", "numpy", "scipy", "scikit-learn"]},
    }


def validate_cases(cases: list[BenchmarkCase]) -> list[dict]:
    records = []
    for case in cases:
        problem = case.create()
        lower, upper = np.asarray(problem.lb), np.asarray(problem.ub)
        assert problem.ndim == case.ndim
        assert lower.shape == upper.shape == (case.ndim,) and np.all(lower < upper)
        optimum = float(problem.f_global)
        point = np.asarray(problem.x_global, dtype=float)
        assert point.shape == (case.ndim,) and np.all((point >= lower) & (point <= upper))
        checked = float(problem.evaluate(point))
        assert np.isclose(checked, optimum, rtol=1e-10, atol=1e-8), (case.name, checked, optimum)
        records.append({**asdict(case), "name": case.name, "dimension_band": case.dimension_band, "known_optimum": optimum, "bounds": np.column_stack((lower, upper)).tolist()})
    return records


class TrackedProblem(Problem):
    def __init__(self, benchmark: Any):
        self.benchmark = benchmark
        self.values: list[float] = []
        self.points: list[list[float]] = []
        super().__init__(n_var=benchmark.ndim, n_obj=1, xl=np.asarray(benchmark.lb), xu=np.asarray(benchmark.ub))

    def _evaluate(self, X, out, *args, **kwargs):
        values = []
        for point in np.asarray(X, dtype=float):
            value = float(self.benchmark.evaluate(point))
            assert np.isfinite(value)
            values.append(value)
            self.values.append(value)
            self.points.append(point.tolist())
        out["F"] = np.asarray(values).reshape(-1, 1)


def _load_previous(root: Path, case: BenchmarkCase, seed: int) -> tuple[dict | None, dict | None]:
    bo_path = root / "bayesian_optimization_highdim_results.json"
    ga_path = root / "bo_vs_ga_equal_budget.json"
    previous_bo = previous_ga = None
    if bo_path.exists():
        data = json.loads(bo_path.read_text())
        if data["settings"]["init_points"] == INITIAL_POINTS and data["settings"]["n_iter"] == BUDGET - INITIAL_POINTS and data["settings"]["kappa"] == 2.576:
            previous_bo = next((r for r in data["runs"] if r["function"] == case.name and r["seed"] == seed), None)
    if ga_path.exists():
        data = json.loads(ga_path.read_text())
        if data["settings"]["evaluations_per_run"] == BUDGET and data["settings"]["ga_pop_size"] == INITIAL_POINTS:
            previous_ga = next((r for r in data["runs"] if r["function"] == case.name and r["seed"] == seed), None)
    return previous_bo, previous_ga


def run_pair(case: BenchmarkCase, seed: int, root_str: str, fingerprint: str, legacy_fingerprint: str | tuple[str, ...] | None = None) -> dict:
    root = Path(root_str)
    cache = root / "optimization_suite_cache" / f"{case.name}_seed{seed}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        accepted_fingerprints = {fingerprint}
        if isinstance(legacy_fingerprint, str):
            accepted_fingerprints.add(legacy_fingerprint)
        elif legacy_fingerprint is not None:
            accepted_fingerprints.update(legacy_fingerprint)
        if saved.get("fingerprint") in accepted_fingerprints:
            saved["fingerprint"] = fingerprint
            return saved
    with threadpool_limits(limits=1):
        benchmark = case.create()
        lower, upper = np.asarray(benchmark.lb), np.asarray(benchmark.ub)
        optimum = float(benchmark.f_global)
        names = [f"x{i}" for i in range(case.ndim)]
        previous_bo, previous_ga = _load_previous(root, case, seed)
        if previous_bo is not None:
            assert previous_bo["ndim"] == case.ndim and previous_bo["evaluations"] == BUDGET
            assert np.allclose(previous_bo["bounds"], np.column_stack((lower, upper)))
            observations = previous_bo["observations"]
            bo_seconds = previous_bo["seconds"]
            bo_source = "previous verified BO experiment"
        else:
            def objective(**parameters: float) -> float:
                unit_point = np.asarray([parameters[key] for key in names])
                value = float(benchmark.evaluate(lower + unit_point * (upper - lower)))
                assert np.isfinite(value)
                return -value

            optimizer = BayesianOptimization(f=objective, pbounds={key: (0.0, 1.0) for key in names}, acquisition_function=UpperConfidenceBound(kappa=2.576), random_state=seed, verbose=0)
            start = perf_counter()
            optimizer.maximize(init_points=INITIAL_POINTS, n_iter=BUDGET - INITIAL_POINTS)
            bo_seconds = perf_counter() - start
            observations = optimizer.res
            assert benchmark.n_fe == BUDGET
            bo_source = "new BO run"
        assert len(observations) == BUDGET
        bo_values = np.asarray([-float(row["target"]) for row in observations])
        bo_points = np.asarray([[row["params"][key] for key in names] for row in observations])
        bo_actual_points = lower + bo_points * (upper - lower)
        shared_initial = bo_actual_points[:INITIAL_POINTS]
        initial_values = bo_values[:INITIAL_POINTS]

        if previous_ga is not None:
            assert previous_ga["evaluations"] == BUDGET and previous_ga["ndim"] == case.ndim
            assert np.allclose(previous_ga["shared_initial_points"], shared_initial)
            ga_values = np.asarray(previous_ga["ga_observed_values"])
            ga_best_x = np.asarray(previous_ga["ga_best_x"])
            ga_seconds = previous_ga["ga_seconds"]
            ga_source = "previous verified shared-initialization GA experiment"
        else:
            ga_benchmark = case.create()
            problem = TrackedProblem(ga_benchmark)
            # pymoo accepts ndarray sampling, though its default-derived static
            # signature suggests only FloatRandomSampling is accepted.
            initial_sampling: Any = shared_initial
            algorithm = GA(pop_size=INITIAL_POINTS, n_offsprings=INITIAL_POINTS, sampling=initial_sampling, eliminate_duplicates=True)
            start = perf_counter()
            result = minimize(problem, algorithm, get_termination("n_eval", BUDGET), seed=seed, verbose=False)
            ga_seconds = perf_counter() - start
            assert result.algorithm is not None and result.X is not None and result.F is not None
            assert result.algorithm.evaluator.n_eval == ga_benchmark.n_fe == BUDGET
            ga_values = np.asarray(problem.values)
            assert np.allclose(problem.points[:INITIAL_POINTS], shared_initial)
            ga_best_x = np.asarray(result.X).ravel()
            assert np.isclose(float(np.asarray(result.F).ravel()[0]), np.min(ga_values))
            ga_source = "new GA run"
        assert len(ga_values) == BUDGET
        assert np.allclose(ga_values[:INITIAL_POINTS], initial_values, rtol=1e-10, atol=1e-10)

        # The reference random search also evaluates all 120 points independently.
        random_benchmark = case.create()
        random_rng = np.random.default_rng(seed + 10000)
        random_points = np.vstack((shared_initial, random_rng.uniform(lower, upper, size=(BUDGET - INITIAL_POINTS, case.ndim))))
        start = perf_counter()
        random_values = np.asarray([float(random_benchmark.evaluate(point)) for point in random_points])
        random_seconds = perf_counter() - start
        assert random_benchmark.n_fe == BUDGET
        assert np.allclose(random_values[:INITIAL_POINTS], initial_values, rtol=1e-10, atol=1e-10)
        assert np.all(np.isfinite(bo_values)) and np.all(np.isfinite(ga_values)) and np.all(np.isfinite(random_values))

        bo_best_index, random_best_index = int(np.argmin(bo_values)), int(np.argmin(random_values))
        best_points = {"bo": bo_actual_points[bo_best_index], "ga": ga_best_x, "random": random_points[random_best_index]}
        observed = {"bo": bo_values, "ga": ga_values, "random": random_values}
        result_record = {
            "fingerprint": fingerprint, "case": case.name, "family": case.family,
            "category": case.category, "dimension_band": case.dimension_band,
            "ndim": case.ndim, "seed": seed, "evaluations": BUDGET,
            "known_optimum": optimum, "initial_best": float(np.min(initial_values)),
            "bo_seconds": bo_seconds, "ga_seconds": ga_seconds, "random_seconds": random_seconds,
            "bo_source": bo_source, "ga_source": ga_source,
            "bo_observations": observations, "shared_initial_points": shared_initial.tolist(),
        }
        for method, values in observed.items():
            best = float(np.min(values))
            best_point = best_points[method]
            assert np.all((best_point >= lower) & (best_point <= upper))
            checker = case.create()
            assert np.isclose(checker.evaluate(best_point), best, rtol=1e-10, atol=1e-8)
            assert best - optimum >= -1e-8 * max(1.0, abs(optimum))
            result_record.update({f"{method}_best": best, f"{method}_gap": best - optimum, f"{method}_best_x": best_point.tolist(), f"{method}_history": np.minimum.accumulate(values).tolist()})
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps(result_record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(cache)
        return result_record


def summarize(cases: list[BenchmarkCase], runs: list[dict]) -> list[dict]:
    summary = []
    for case in cases:
        case_runs = [row for row in runs if row["case"] == case.name]
        assert len(case_runs) == len(SEEDS)
        optimum = case_runs[0]["known_optimum"]
        initial_gap = float(np.median([row["initial_best"] - optimum for row in case_runs]))
        row = {"case": case.name, "family": case.family, "category": case.category, "ndim": case.ndim, "dimension_band": case.dimension_band, "evaluations": BUDGET, "known_optimum": optimum, "initial_gap_median": initial_gap}
        for method in ["bo", "ga", "random"]:
            values = [r[f"{method}_best"] for r in case_runs]
            median = float(np.median(values))
            gap = median - optimum
            relative = float(np.median([max(0.0, r[f"{method}_gap"]) / max(1e-12, r["initial_best"] - optimum) for r in case_runs]))
            row.update({f"{method}_median": median, f"{method}_gap_median": gap, f"{method}_best": min(values), f"{method}_worst": max(values), f"{method}_relative_gap": relative, f"{method}_seconds_median": float(np.median([r[f"{method}_seconds"] for r in case_runs]))})
        tolerance = 1e-10 * max(1.0, initial_gap)
        row["winner"] = "Tie" if abs(row["bo_median"] - row["ga_median"]) <= tolerance else "BO" if row["bo_median"] < row["ga_median"] else "GA"
        row["ga_paired_wins"] = sum(r["ga_best"] < r["bo_best"] for r in case_runs)
        summary.append(row)
    return summary


def grouped_summary(summary: list[dict], key: str) -> list[dict]:
    groups = []
    for label in sorted({row[key] for row in summary}):
        rows = [row for row in summary if row[key] == label]
        groups.append({"group_by": key, "group": label, "cases": len(rows), "bo_wins": sum(row["winner"] == "BO" for row in rows), "ga_wins": sum(row["winner"] == "GA" for row in rows), "ties": sum(row["winner"] == "Tie" for row in rows), "bo_median_relative_gap": float(np.median([row["bo_relative_gap"] for row in rows])), "ga_median_relative_gap": float(np.median([row["ga_relative_gap"] for row in rows])), "random_median_relative_gap": float(np.median([row["random_relative_gap"] for row in rows]))})
    return groups


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(output_dir: str = "data", workers: int = 4) -> dict:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    specs = validate_cases(cases)
    config = configuration()
    fingerprint = hashlib.sha256(json.dumps({"config": config, "cases": specs}, sort_keys=True).encode()).hexdigest()
    # The initial 56-case batch remains valid when adding the four discontinuous
    # cases: its settings and function specifications are identical.
    legacy_specs = [dict(spec) for spec in specs if spec["category"] != "discontinuous"]
    # The first batch used F10. Its installed implementation violates the claimed
    # optimum, so F6 replaces it. Other cached cases retain their identical setup.
    for spec in legacy_specs:
        if spec["family"] == "CEC_F6":
            spec.update(family="CEC_F10", classname="F102014", category="multimodal_shifted", description="Shifted Schwefel; boundary structure", name="CEC_F10_10D", known_optimum=1000.0)
        elif spec["family"] == "CEC_F19":
            spec.update(family="CEC_F17", classname="F172014", description="Shuffled, rotated variable groups; hybrid subfunctions", name=f"CEC_F17_{spec['ndim']}D", known_optimum=1700.0)
    legacy_fingerprint = hashlib.sha256(json.dumps({"config": config, "cases": legacy_specs}, sort_keys=True).encode()).hexdigest()
    legacy_full_specs = legacy_specs + [spec for spec in specs if spec["category"] == "discontinuous"]
    legacy_full_fingerprint = hashlib.sha256(json.dumps({"config": config, "cases": legacy_full_specs}, sort_keys=True).encode()).hexdigest()
    intermediate_specs = [dict(spec) for spec in specs]
    for spec in intermediate_specs:
        if spec["family"] == "CEC_F19":
            spec.update(family="CEC_F17", classname="F172014", description="Shuffled, rotated variable groups; hybrid subfunctions", name=f"CEC_F17_{spec['ndim']}D", known_optimum=1700.0)
    intermediate_fingerprint = hashlib.sha256(json.dumps({"config": config, "cases": intermediate_specs}, sort_keys=True).encode()).hexdigest()
    jobs = [(case, seed, str(root), fingerprint, (legacy_fingerprint, legacy_full_fingerprint, intermediate_fingerprint)) for case in cases for seed in SEEDS]
    runs = []
    print(f"{len(cases)} cases, {len(jobs)} paired runs, budget={BUDGET}, workers={workers}", flush=True)
    # Spawned processes isolate each algorithm's random state. These are Python
    # computation workers, not agents; BLAS is limited to one thread per worker.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_pair, *job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            runs.append(row)
            print(f"[{completed:3d}/{len(jobs)}] {row['case']:<20} seed={row['seed']:2d} BO={row['bo_best']:.7g} GA={row['ga_best']:.7g} ({row['bo_seconds']:.1f}s BO)", flush=True)
    order = {case.name: i for i, case in enumerate(cases)}
    runs.sort(key=lambda row: (order[row["case"]], row["seed"]))
    summary = summarize(cases, runs)
    groups = grouped_summary(summary, "dimension_band") + grouped_summary(summary, "category")
    experiment = {"configuration": config, "fingerprint": fingerprint, "cases": specs, "summary": summary, "groups": groups, "runs": runs}
    path = root / "optimization_suite_results.json"
    path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n")
    _write_csv(root / "optimization_suite_summary.csv", summary)
    _write_csv(root / "optimization_suite_groups.csv", groups)
    flat_runs = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in runs]
    _write_csv(root / "optimization_suite_runs.csv", flat_runs)
    print(f"Saved {path}", flush=True)
    return experiment


if __name__ == "__main__":
    # Thread limits are inherited by spawned workers and avoid BLAS oversubscription.
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[key] = "1"
    run_suite()
