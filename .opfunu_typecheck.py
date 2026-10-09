import inspect
from typing import cast
import numpy as np
import opfunu
from opfunu.name_based import Ackley01

np.set_printoptions(precision=4, suppress=True)
func = Ackley01(ndim=3)
# opfunu 的基类先将维度设为 None；构造后检查并保存整数维度。
raw_ndim = func.ndim
assert isinstance(raw_ndim, int)
ndim: int = raw_ndim
print("opfunu 版本:", opfunu.__version__)
print("构造函数:", inspect.signature(Ackley01))
print("函数名称:", func.name)
print("维度:", func.ndim)
print("每一维的下界:", func.lb)
print("每一维的上界:", func.ub)
print("bounds（每行是该维度的 [下界, 上界]）:\n", func.bounds)
print("已知最优解:", func.x_global)
print("已知最优值:", func.f_global)

x = np.array([1.0, 2.0, 3.0])
assert x.shape == (func.ndim,)
assert np.all((x >= func.lb) & (x <= func.ub))

# 明确这是运行时整数，避免 notebook 中被误判为类型别名。
evaluation_count_before: int = func.n_fe
value = func.evaluate(x)
print("候选解:", x)
print("目标值:", value)
print("本次求值次数:", func.n_fe - evaluation_count_before)

opt_value = func.evaluate(func.x_global)
print("最优解处的目标值:", opt_value)
print("是否达到最优值:", func.is_succeed(func.x_global, tol=1e-5))
print("累计求值次数（含 is_succeed 内部求值）:", func.n_fe)
assert np.isclose(opt_value, func.f_global, atol=1e-10)

# create_solution 使用 NumPy 全局随机状态；种子使这一演示可重复。
np.random.seed(42)
x_random = func.create_solution()
print("随机候选解:", x_random)
print("随机候选解的目标值:", func.evaluate(x_random))
print("函数参数:", func.get_paras())

rng = np.random.default_rng(42)
population = rng.uniform(func.lb, func.ub, size=(1000, ndim))
search_count_before: int = func.n_fe
scores = np.array([func.evaluate(candidate) for candidate in population])
best_index = np.argmin(scores)
best_x = population[best_index]
best_value = scores[best_index]

print("随机搜索的最好解:", best_x)
print("最好目标值:", best_value)
print("与已知最优值的差:", best_value - func.f_global)
print("本次搜索的求值次数:", func.n_fe - search_count_before)
assert func.n_fe - search_count_before == len(population)

from opfunu.cec_based import F12014

cec_func = F12014(ndim=10)
print("CEC 函数名称:", cec_func.name)
print("支持的维度:", cec_func.dim_supported)
# 边界在基类中初始化为 None；先检查，再使用下标。
cec_bounds = cec_func.bounds
assert cec_bounds is not None
print("搜索边界（第 1 维）:", cec_bounds[0])
print("已知最优值:", cec_func.f_global)
print("已知最优解处的目标值:", cec_func.evaluate(cec_func.x_global))
assert np.isclose(cec_func.evaluate(cec_func.x_global), cec_func.f_global)

classes = opfunu.get_functions_by_classname("Ackley01")
print("按完整类名查找:", [cls.__name__ for cls in classes])
# 查找返回动态类列表，静态分析无法知道这里是 Ackley01。
assert classes and classes[0] is Ackley01
found_class = cast(type[Ackley01], classes[0])
found_func = found_class(ndim=3)
print("查找后实例化并求值:", found_func.evaluate(np.zeros(3)))

cec_2014_classes = opfunu.get_functions_based_classname("2014")
print("名称含 2014 的函数数量:", len(cec_2014_classes))
print("前 5 个函数:", [cls.__name__ for cls in cec_2014_classes[:5]])

# 查看具体函数的签名和说明：
print("evaluate 签名:", inspect.signature(func.evaluate))
# help(Ackley01)
# help(func.evaluate)
# dir(func)

from importlib.metadata import version
from time import perf_counter
from typing import Callable
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound
from opfunu.benchmark import Benchmark
from opfunu.name_based.a_func import Ackley01
from opfunu.name_based.b_func import Branin01
from opfunu.name_based.g_func import Griewank
from opfunu.name_based.l_func import Levy03

print("bayesian-optimization 版本:", version("bayesian-optimization"))
BO_INIT_POINTS: int = 10
BO_N_ITER: int = 50
BO_SEEDS = [0, 1, 42]
BO_CASES: list[tuple[str, Callable[[], Benchmark]]] = [
    ("Ackley01", lambda: Ackley01(ndim=2)),
    ("Griewank", lambda: Griewank(ndim=2)),
    ("Levy03", lambda: Levy03(ndim=2)),
    ("Branin01", lambda: Branin01()),
]


def run_bo_case(
    name: str, factory: Callable[[], Benchmark], seed: int,
    init_points: int = BO_INIT_POINTS, n_iter: int = BO_N_ITER,
) -> dict:
    problem = factory()
    problem_ndim = problem.ndim
    assert isinstance(problem_ndim, int) and problem_ndim > 0
    lower = np.asarray(problem.lb, dtype=float)
    upper = np.asarray(problem.ub, dtype=float)
    assert np.all(upper > lower)
    optimum = problem.f_global
    assert optimum is not None
    optimum_value = float(optimum)
    parameter_names = [f"x{i}" for i in range(problem_ndim)]

    def objective(**params: float) -> float:
        unit_x = np.array([params[key] for key in parameter_names])
        actual_x = lower + unit_x * (upper - lower)
        return -float(problem.evaluate(actual_x))

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={key: (0.0, 1.0) for key in parameter_names},
        acquisition_function=UpperConfidenceBound(kappa=2.576),
        random_state=seed,
        verbose=0,
    )
    start = perf_counter()
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    elapsed = perf_counter() - start
    observations = optimizer.res
    observed_values = np.array([-float(row["target"]) for row in observations])
    best_index = int(np.argmin(observed_values))
    best_params = observations[best_index]["params"]
    best_unit_x = np.array([best_params[key] for key in parameter_names])
    best_actual_x = lower + best_unit_x * (upper - lower)
    best_value = float(observed_values[best_index])
    assert len(observations) == init_points + n_iter
    assert problem.n_fe == len(observations)
    assert np.all((best_actual_x >= lower) & (best_actual_x <= upper))

    # 使用独立对象做随机搜索和结果复核，不计入 BO 的评估预算。
    random_problem = factory()
    random_rng = np.random.default_rng(seed)
    random_population = random_rng.uniform(lower, upper, size=(len(observations), problem_ndim))
    random_values = np.array([float(random_problem.evaluate(x)) for x in random_population])
    checked_value = float(random_problem.evaluate(best_actual_x))
    assert np.isclose(checked_value, best_value, rtol=1e-10, atol=1e-10)
    return {
        "function": name, "ndim": problem_ndim, "seed": seed,
        "evaluations": len(observations), "best_value": best_value,
        "known_optimum": optimum_value, "gap": best_value - optimum_value,
        "best_x": best_actual_x.tolist(),
        "bounds": np.column_stack((lower, upper)).tolist(),
        "random_best": float(np.min(random_values)), "seconds": elapsed,
        "best_history": np.minimum.accumulate(observed_values).tolist(),
        "random_history": np.minimum.accumulate(random_values).tolist(),
        "observations": observations,
    }

bo_results: list[dict] = []
for case_name, case_factory in BO_CASES:
    for run_seed in BO_SEEDS:
        run_result = run_bo_case(case_name, case_factory, run_seed)
        bo_results.append(run_result)
        print(
            f"{case_name:10s} seed={run_seed:2d} "
            f"best={run_result['best_value']:.9g} "
            f"gap={run_result['gap']:.6g} "
            f"random={run_result['random_best']:.6g} "
            f"time={run_result['seconds']:.1f}s",
            flush=True,
        )

print("每个函数 3 次运行的汇总（目标值越小越好）")
print(f"{'Function':<12} {'Known optimum':>14} {'BO best':>14} {'BO median':>14} {'BO worst':>14} {'Random median':>14}")
bo_summary: list[dict] = []
for case_name, _ in BO_CASES:
    case_runs = [row for row in bo_results if row["function"] == case_name]
    winner = min(case_runs, key=lambda row: row["best_value"])
    median_value = float(np.median([row["best_value"] for row in case_runs]))
    worst_value = max(row["best_value"] for row in case_runs)
    random_median = float(np.median([row["random_best"] for row in case_runs]))
    print(f"{case_name:<12} {winner['known_optimum']:>14.9g} {winner['best_value']:>14.9g} {median_value:>14.9g} {worst_value:>14.9g} {random_median:>14.9g}")
    print(f"  最好解={winner['best_x']}，种子={winner['seed']}，距已知最优值={winner['gap']:.9g}")
    bo_summary.append({
        "function": case_name, "known_optimum": winner["known_optimum"],
        "best_value": winner["best_value"], "median_value": median_value,
        "worst_value": worst_value, "best_seed": winner["seed"],
        "best_x": winner["best_x"], "gap": winner["gap"],
        "random_median": random_median,
    })

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for axis, (case_name, _) in zip(axes.flat, BO_CASES):
    case_runs = [row for row in bo_results if row["function"] == case_name]
    known_value = case_runs[0]["known_optimum"]
    bo_gaps = np.array([row["best_history"] for row in case_runs]) - known_value
    random_gaps = np.array([row["random_history"] for row in case_runs]) - known_value
    counts = np.arange(1, BO_INIT_POINTS + BO_N_ITER + 1)
    axis.plot(counts, np.maximum(np.median(bo_gaps, axis=0), 1e-12), label="BO median gap")
    axis.fill_between(
        counts, np.maximum(np.min(bo_gaps, axis=0), 1e-12),
        np.maximum(np.max(bo_gaps, axis=0), 1e-12), alpha=0.2,
    )
    axis.plot(counts, np.maximum(np.median(random_gaps, axis=0), 1e-12), "--", label="Random median gap")
    axis.axvline(BO_INIT_POINTS, color="gray", linestyle=":", label="End of initialization")
    axis.set(title=case_name, xlabel="Function evaluations", ylabel="Best value - known optimum", yscale="log")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
fig.suptitle("Bayesian optimization: 2D, full default bounds, 3 seeds")
fig.tight_layout()
plt.show()

import csv
import json
from pathlib import Path

result_dir = Path("data")
result_dir.mkdir(exist_ok=True)
experiment = {
    "versions": {name: version(name) for name in ["bayesian-optimization", "opfunu", "numpy", "scipy", "scikit-learn"]},
    "settings": {"init_points": BO_INIT_POINTS, "n_iter": BO_N_ITER, "seeds": BO_SEEDS, "acquisition": "UCB", "kappa": 2.576, "input_scaling": "unit cube"},
    "summary": bo_summary, "runs": bo_results,
}
json_path = result_dir / "bayesian_optimization_results.json"
json_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n")
csv_path = result_dir / "bayesian_optimization_results.csv"
fields = ["function", "ndim", "seed", "evaluations", "best_value", "known_optimum", "gap", "best_x", "random_best", "seconds"]
with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(bo_results)
print("详细结果、参数及全部采样记录:", json_path)
print("每次运行的结果表:", csv_path)

from opfunu.cec_based.cec2014 import F32014, F92014, F172014

HIGH_INIT_POINTS: int = 20
HIGH_N_ITER: int = 100
HIGH_SEEDS = [0, 1, 42]
HIGH_CASES: list[tuple[str, Callable[[], Benchmark]]] = [
    ("Ackley01_5D", lambda: Ackley01(ndim=5)),
    ("Ackley01_10D", lambda: Ackley01(ndim=10)),
    ("Ackley01_20D", lambda: Ackley01(ndim=20)),
    ("Griewank_10D", lambda: Griewank(ndim=10)),
    ("CEC_F3_10D", lambda: F32014(ndim=10)),
    ("CEC_F9_10D", lambda: F92014(ndim=10)),
    ("CEC_F9_20D", lambda: F92014(ndim=20)),
    ("CEC_F17_10D", lambda: F172014(ndim=10)),
]

# 在独立对象上验证基准最优值，不占用优化预算。
for high_name, high_factory in HIGH_CASES:
    reference_problem = high_factory()
    reference_optimum = reference_problem.f_global
    reference_x = reference_problem.x_global
    assert reference_optimum is not None and reference_x is not None
    assert np.isclose(reference_problem.evaluate(reference_x), reference_optimum, atol=1e-8)
    print(f"{high_name:<14} ndim={reference_problem.ndim}, optimum={reference_optimum}, bounds[0]={[reference_problem.lb[0], reference_problem.ub[0]]}")

high_results: list[dict] = []
for high_name, high_factory in HIGH_CASES:
    for high_seed in HIGH_SEEDS:
        high_result = run_bo_case(
            high_name, high_factory, high_seed,
            init_points=HIGH_INIT_POINTS, n_iter=HIGH_N_ITER,
        )
        high_results.append(high_result)
        print(
            f"{high_name:<14} seed={high_seed:2d} "
            f"best={high_result['best_value']:.9g} "
            f"gap={high_result['gap']:.6g} "
            f"random={high_result['random_best']:.6g} "
            f"time={high_result['seconds']:.1f}s",
            flush=True,
        )

high_summary: list[dict] = []
print("高维实验：每组 3 个种子，每次 120 次评估，越小越好")
print(f"{'Case':<15} {'Optimum':>12} {'BO best':>14} {'Best gap':>14} {'BO median':>14} {'Random median':>14} {'Init median':>14}")
for high_name, _ in HIGH_CASES:
    high_runs = [row for row in high_results if row["function"] == high_name]
    high_winner = min(high_runs, key=lambda row: row["best_value"])
    high_median = float(np.median([row["best_value"] for row in high_runs]))
    high_random_median = float(np.median([row["random_best"] for row in high_runs]))
    high_initial_median = float(np.median([row["best_history"][HIGH_INIT_POINTS - 1] for row in high_runs]))
    high_summary.append({
        "function": high_name, "ndim": high_winner["ndim"],
        "known_optimum": high_winner["known_optimum"],
        "best_value": high_winner["best_value"], "gap": high_winner["gap"],
        "median_value": high_median,
        "worst_value": max(row["best_value"] for row in high_runs),
        "random_median": high_random_median, "initial_median": high_initial_median,
        "best_seed": high_winner["seed"], "best_x": high_winner["best_x"],
    })
    print(f"{high_name:<15} {high_winner['known_optimum']:>12.6g} {high_winner['best_value']:>14.9g} {high_winner['gap']:>14.9g} {high_median:>14.9g} {high_random_median:>14.9g} {high_initial_median:>14.9g}")
    print(f"  最好解（seed={high_winner['seed']}）: {high_winner['best_x']}")

print("\nInit median 是 BO 前 20 次随机初始化结束时的最好值中位数，便于观察后续 100 次迭代带来的改进。")

import matplotlib.pyplot as plt

high_fig, high_axes = plt.subplots(4, 2, figsize=(13, 15))
for high_axis, (high_name, _) in zip(high_axes.flat, HIGH_CASES):
    high_runs = [row for row in high_results if row["function"] == high_name]
    high_optimum = high_runs[0]["known_optimum"]
    high_bo_gaps = np.array([row["best_history"] for row in high_runs]) - high_optimum
    high_random_gaps = np.array([row["random_history"] for row in high_runs]) - high_optimum
    high_counts = np.arange(1, HIGH_INIT_POINTS + HIGH_N_ITER + 1)
    high_axis.plot(high_counts, np.maximum(np.median(high_bo_gaps, axis=0), 1e-12), label="BO median gap")
    high_axis.fill_between(
        high_counts, np.maximum(np.min(high_bo_gaps, axis=0), 1e-12),
        np.maximum(np.max(high_bo_gaps, axis=0), 1e-12), alpha=0.2,
    )
    high_axis.plot(high_counts, np.maximum(np.median(high_random_gaps, axis=0), 1e-12), "--", label="Random median gap")
    high_axis.axvline(HIGH_INIT_POINTS, color="gray", linestyle=":")
    high_axis.set(title=high_name, xlabel="Function evaluations", ylabel="Best value - known optimum", yscale="log")
    high_axis.grid(alpha=0.25)
    high_axis.legend(fontsize=8)
high_fig.suptitle("Higher dimensions and CEC functions: 120 evaluations, 3 seeds")
high_fig.tight_layout()
plt.show()

import csv
import json
from pathlib import Path

high_result_dir = Path("data")
high_result_dir.mkdir(exist_ok=True)
high_experiment = {
    "versions": {name: version(name) for name in ["bayesian-optimization", "opfunu", "numpy", "scipy", "scikit-learn"]},
    "settings": {"init_points": HIGH_INIT_POINTS, "n_iter": HIGH_N_ITER, "seeds": HIGH_SEEDS, "acquisition": "UCB", "kappa": 2.576, "input_scaling": "unit cube"},
    "summary": high_summary, "runs": high_results,
}
high_json_path = high_result_dir / "bayesian_optimization_highdim_results.json"
high_json_path.write_text(json.dumps(high_experiment, ensure_ascii=False, indent=2) + "\n")
high_csv_path = high_result_dir / "bayesian_optimization_highdim_results.csv"
high_fields = ["function", "ndim", "seed", "evaluations", "best_value", "known_optimum", "gap", "best_x", "random_best", "seconds"]
with high_csv_path.open("w", newline="", encoding="utf-8") as high_csv_file:
    high_writer = csv.DictWriter(high_csv_file, fieldnames=high_fields, extrasaction="ignore")
    high_writer.writeheader()
    high_writer.writerows(high_results)
print("高维实验完整记录:", high_json_path)
print("高维实验结果表:", high_csv_path)

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import json
from pathlib import Path

GA_POP_SIZE: int = 40
GA_MAX_EVALS: int = 6000
GA_CHECKPOINTS = [120, 1200, 6000]
GA_SEEDS = [0, 1, 42]
print("pymoo 版本:", version("pymoo"))


class OpfunuGAProblem(Problem):
    """将 opfunu 函数接入 pymoo，并记录实际目标求值。"""

    def __init__(self, benchmark: Benchmark):
        problem_dimension = benchmark.ndim
        assert isinstance(problem_dimension, int) and problem_dimension > 0
        self.benchmark = benchmark
        self.evaluated_values: list[float] = []
        self.evaluated_points: list[list[float]] = []
        super().__init__(
            n_var=problem_dimension, n_obj=1,
            xl=np.asarray(benchmark.lb, dtype=float),
            xu=np.asarray(benchmark.ub, dtype=float),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        candidates = np.asarray(X, dtype=float)
        values = []
        for candidate in candidates:
            value = float(self.benchmark.evaluate(candidate))
            values.append(value)
            self.evaluated_values.append(value)
            self.evaluated_points.append(candidate.tolist())
        out["F"] = np.asarray(values, dtype=float).reshape(-1, 1)


def run_ga_case(name: str, factory: Callable[[], Benchmark], seed: int) -> dict:
    benchmark = factory()
    optimum = benchmark.f_global
    assert optimum is not None
    optimum_value = float(optimum)
    problem = OpfunuGAProblem(benchmark)
    ga_algorithm = GA(
        pop_size=GA_POP_SIZE, n_offsprings=GA_POP_SIZE,
        eliminate_duplicates=True,
    )
    start = perf_counter()
    ga_result = minimize(
        problem, ga_algorithm, get_termination("n_eval", GA_MAX_EVALS),
        seed=seed, verbose=False,
    )
    elapsed = perf_counter() - start
    assert ga_result.algorithm is not None
    actual_evaluations = int(ga_result.algorithm.evaluator.n_eval)
    assert actual_evaluations == GA_MAX_EVALS
    assert len(problem.evaluated_values) == actual_evaluations
    assert benchmark.n_fe == actual_evaluations
    assert ga_result.F is not None and ga_result.X is not None
    final_value = float(np.asarray(ga_result.F).ravel()[0])
    final_x = np.asarray(ga_result.X, dtype=float).ravel()
    assert np.all((final_x >= benchmark.lb) & (final_x <= benchmark.ub))
    # 使用独立基准对象复核，避免增加本次算法的评估次数。
    checker = factory()
    assert np.isclose(checker.evaluate(final_x), final_value, rtol=1e-10, atol=1e-10)

    values = np.asarray(problem.evaluated_values)
    best_history = np.minimum.accumulate(values)
    checkpoint_rows: list[dict] = []
    for checkpoint in GA_CHECKPOINTS:
        index = int(np.argmin(values[:checkpoint]))
        checkpoint_value = float(values[index])
        checkpoint_rows.append({
            "function": name, "ndim": int(problem.n_var), "seed": seed,
            "evaluations": checkpoint, "best_value": checkpoint_value,
            "known_optimum": optimum_value, "gap": checkpoint_value - optimum_value,
            "best_x": problem.evaluated_points[index],
        })
    assert np.isclose(final_value, checkpoint_rows[-1]["best_value"])
    # GA 每一代是一批候选解，只在完整代末记录收敛曲线。
    generation_counts = np.arange(GA_POP_SIZE, actual_evaluations + 1, GA_POP_SIZE)
    return {
        "function": name, "ndim": int(problem.n_var), "seed": seed,
        "evaluations": actual_evaluations, "seconds": elapsed,
        "known_optimum": optimum_value, "best_value": final_value,
        "gap": final_value - optimum_value, "best_x": final_x.tolist(),
        "checkpoints": checkpoint_rows,
        "history_evaluations": generation_counts.tolist(),
        "best_history": best_history[generation_counts - 1].tolist(),
    }


bo_comparison_path = Path("data/bayesian_optimization_highdim_results.json")
bo_comparison = json.loads(bo_comparison_path.read_text())
assert bo_comparison["settings"]["init_points"] + bo_comparison["settings"]["n_iter"] == 120
assert bo_comparison["settings"]["seeds"] == GA_SEEDS
print("BO 比较数据:", bo_comparison_path)

ga_results: list[dict] = []
for ga_name, ga_factory in HIGH_CASES:
    for ga_seed in GA_SEEDS:
        ga_run = run_ga_case(ga_name, ga_factory, ga_seed)
        ga_results.append(ga_run)
        checkpoint_text = ", ".join(
            f"{row['evaluations']}次={row['best_value']:.9g}"
            for row in ga_run["checkpoints"]
        )
        print(f"{ga_name:<14} seed={ga_seed:2d} {checkpoint_text}, time={ga_run['seconds']:.2f}s", flush=True)

ga_summary: list[dict] = []
print("三次运行的中位数对比（目标越小越好）")
print(f"{'Case':<15} {'BO@120':>14} {'GA@120':>14} {'GA@1200':>14} {'GA@6000':>14} {'GA best@6000':>16} {'Best gap':>14}")
for ga_name, _ in HIGH_CASES:
    ga_case_runs = [row for row in ga_results if row["function"] == ga_name]
    bo_case_runs = [row for row in bo_comparison["runs"] if row["function"] == ga_name]
    assert len(ga_case_runs) == len(bo_case_runs) == len(GA_SEEDS)
    assert all(row["evaluations"] == 120 for row in bo_case_runs)
    bo_median = float(np.median([row["best_value"] for row in bo_case_runs]))
    ga_medians = {
        checkpoint: float(np.median([
            next(item["best_value"] for item in row["checkpoints"] if item["evaluations"] == checkpoint)
            for row in ga_case_runs
        ]))
        for checkpoint in GA_CHECKPOINTS
    }
    ga_winner = min(ga_case_runs, key=lambda row: row["best_value"])
    ga_summary.append({
        "function": ga_name, "ndim": ga_winner["ndim"],
        "known_optimum": ga_winner["known_optimum"],
        "bo_median_120": bo_median,
        "ga_median_120": ga_medians[120], "ga_median_1200": ga_medians[1200],
        "ga_median_6000": ga_medians[6000], "ga_best_6000": ga_winner["best_value"],
        "ga_best_gap_6000": ga_winner["gap"], "ga_best_seed": ga_winner["seed"],
        "ga_best_x": ga_winner["best_x"],
    })
    print(f"{ga_name:<15} {bo_median:>14.9g} {ga_medians[120]:>14.9g} {ga_medians[1200]:>14.9g} {ga_medians[6000]:>14.9g} {ga_winner['best_value']:>16.9g} {ga_winner['gap']:>14.9g}")

ga_same_budget_wins = sum(row["ga_median_120"] < row["bo_median_120"] for row in ga_summary)
print(f"\n同预算 120 次时，GA 中位数更好的组数: {ga_same_budget_wins}/{len(ga_summary)}")
print("6000 次 GA 与 120 次 BO 的预算不同，不能据此判定算法优劣。")

import matplotlib.pyplot as plt

ga_fig, ga_axes = plt.subplots(4, 2, figsize=(13, 15))
for ga_axis, (ga_name, _) in zip(ga_axes.flat, HIGH_CASES):
    ga_case_runs = [row for row in ga_results if row["function"] == ga_name]
    bo_case_runs = [row for row in bo_comparison["runs"] if row["function"] == ga_name]
    ga_known_value = ga_case_runs[0]["known_optimum"]
    ga_gap_histories = np.asarray([row["best_history"] for row in ga_case_runs]) - ga_known_value
    ga_eval_counts = np.asarray(ga_case_runs[0]["history_evaluations"])
    ga_axis.plot(ga_eval_counts, np.maximum(np.median(ga_gap_histories, axis=0), 1e-12), label="GA median gap")
    ga_axis.fill_between(
        ga_eval_counts, np.maximum(np.min(ga_gap_histories, axis=0), 1e-12),
        np.maximum(np.max(ga_gap_histories, axis=0), 1e-12), alpha=0.2,
    )
    bo_gap_histories = np.asarray([row["best_history"] for row in bo_case_runs]) - ga_known_value
    ga_axis.plot(
        np.arange(1, 121), np.maximum(np.median(bo_gap_histories, axis=0), 1e-12),
        "--", label="BO median gap (ends at 120)",
    )
    ga_axis.axvline(120, color="gray", linestyle=":")
    ga_axis.set(title=ga_name, xlabel="Function evaluations", ylabel="Best value - known optimum", xscale="log", yscale="log")
    ga_axis.grid(alpha=0.25)
    ga_axis.legend(fontsize=8)
ga_fig.suptitle("pymoo GA: population 40, 6000 evaluations, 3 seeds")
ga_fig.tight_layout()
plt.show()

import csv
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

def ga_scalar_parameter(value: object) -> float:
    # pymoo 参数容器的 value 静态类型很宽；验证后转换为可保存的标量。
    assert isinstance(value, (int, float, np.integer, np.floating))
    return float(value)


# 读取实际默认算子值，而不是凭文档假定参数。
ga_default_algorithm = GA(pop_size=GA_POP_SIZE, n_offsprings=GA_POP_SIZE)
ga_crossover = ga_default_algorithm.mating.crossover
ga_mutation = ga_default_algorithm.mating.mutation
assert isinstance(ga_crossover, SBX) and isinstance(ga_mutation, PM)
ga_experiment = {
    "versions": {name: version(name) for name in ["pymoo", "opfunu", "numpy", "scipy"]},
    "settings": {
        "pop_size": GA_POP_SIZE, "n_offsprings": GA_POP_SIZE,
        "max_evaluations": GA_MAX_EVALS, "checkpoints": GA_CHECKPOINTS,
        "seeds": GA_SEEDS, "eliminate_duplicates": True,
        "crossover": "SBX", "crossover_probability": ga_scalar_parameter(ga_crossover.prob.value),
        "crossover_eta": ga_scalar_parameter(ga_crossover.eta.value),
        "mutation": "PM", "mutation_probability": ga_scalar_parameter(ga_mutation.prob.value),
        "mutation_eta": ga_scalar_parameter(ga_mutation.eta.value),
        "mutation_probability_per_variable": "pymoo default: min(0.5, 1 / ndim)",
        "bounds": "opfunu full default bounds", "objective": "minimize",
        "bo_comparison_file": str(bo_comparison_path),
    },
    "summary": ga_summary, "runs": ga_results,
}
ga_result_dir = Path("data")
ga_result_dir.mkdir(exist_ok=True)
ga_json_path = ga_result_dir / "pymoo_ga_results.json"
ga_json_path.write_text(json.dumps(ga_experiment, ensure_ascii=False, indent=2) + "\n")
ga_csv_path = ga_result_dir / "pymoo_ga_results.csv"
ga_fields = ["function", "ndim", "seed", "evaluations", "best_value", "known_optimum", "gap", "best_x"]
with ga_csv_path.open("w", newline="", encoding="utf-8") as ga_csv_file:
    ga_writer = csv.DictWriter(ga_csv_file, fieldnames=ga_fields)
    ga_writer.writeheader()
    for ga_run in ga_results:
        ga_writer.writerows(ga_run["checkpoints"])
print("GA 参数与完整结果:", ga_json_path)
print("120 / 1200 / 6000 次评估结果表:", ga_csv_path)

EQUAL_BUDGET: int = 120
EQUAL_POP_SIZE: int = 20
EQUAL_SEEDS = [0, 1, 42]
equal_bo_data = json.loads(Path("data/bayesian_optimization_highdim_results.json").read_text())
assert equal_bo_data["settings"]["init_points"] == EQUAL_POP_SIZE
assert equal_bo_data["settings"]["init_points"] + equal_bo_data["settings"]["n_iter"] == EQUAL_BUDGET
assert equal_bo_data["settings"]["seeds"] == EQUAL_SEEDS


def run_equal_ga(name: str, factory: Callable[[], Benchmark], seed: int) -> dict:
    bo_runs = [row for row in equal_bo_data["runs"] if row["function"] == name and row["seed"] == seed]
    assert len(bo_runs) == 1
    bo_run = bo_runs[0]
    assert bo_run["evaluations"] == len(bo_run["observations"]) == EQUAL_BUDGET
    benchmark = factory()
    dimension = benchmark.ndim
    assert isinstance(dimension, int) and dimension == bo_run["ndim"]
    lower = np.asarray(benchmark.lb, dtype=float)
    upper = np.asarray(benchmark.ub, dtype=float)
    assert np.allclose(np.column_stack((lower, upper)), bo_run["bounds"])
    parameter_names = [f"x{i}" for i in range(dimension)]
    bo_initial_records = bo_run["observations"][:EQUAL_POP_SIZE]
    unit_initial = np.asarray([
        [row["params"][key] for key in parameter_names]
        for row in bo_initial_records
    ], dtype=float)
    shared_initial = lower + unit_initial * (upper - lower)
    bo_initial_values = np.asarray([-float(row["target"]) for row in bo_initial_records])
    problem = OpfunuGAProblem(benchmark)
    equal_algorithm = GA(
        pop_size=EQUAL_POP_SIZE, n_offsprings=EQUAL_POP_SIZE,
        sampling=shared_initial, eliminate_duplicates=True,
    )
    start = perf_counter()
    result = minimize(
        problem, equal_algorithm, get_termination("n_eval", EQUAL_BUDGET),
        seed=seed, verbose=False,
    )
    elapsed = perf_counter() - start
    assert result.algorithm is not None and result.X is not None and result.F is not None
    evaluations = int(result.algorithm.evaluator.n_eval)
    assert evaluations == benchmark.n_fe == len(problem.evaluated_values) == EQUAL_BUDGET
    values = np.asarray(problem.evaluated_values)
    assert np.allclose(values[:EQUAL_POP_SIZE], bo_initial_values, rtol=1e-10, atol=1e-10)
    assert np.allclose(np.asarray(problem.evaluated_points[:EQUAL_POP_SIZE]), shared_initial)
    optimum = benchmark.f_global
    assert optimum is not None and np.isclose(optimum, bo_run["known_optimum"])
    best_value = float(np.asarray(result.F).ravel()[0])
    best_x = np.asarray(result.X, dtype=float).ravel()
    assert np.isclose(best_value, np.min(values))
    assert np.all((best_x >= lower) & (best_x <= upper))
    checker = factory()
    assert np.isclose(checker.evaluate(best_x), best_value, rtol=1e-10, atol=1e-10)
    return {
        "function": name, "ndim": dimension, "seed": seed,
        "evaluations": evaluations, "known_optimum": float(optimum),
        "bo_best_value": bo_run["best_value"], "ga_best_value": best_value,
        "bo_gap": bo_run["gap"], "ga_gap": best_value - float(optimum),
        "bo_best_x": bo_run["best_x"], "ga_best_x": best_x.tolist(),
        "shared_initial_best": float(np.min(bo_initial_values)),
        "shared_initial_points": shared_initial.tolist(),
        "bo_seconds": bo_run["seconds"], "ga_seconds": elapsed,
        "bo_best_history": bo_run["best_history"],
        "ga_best_history": np.minimum.accumulate(values).tolist(),
        "ga_observed_values": values.tolist(),
    }

equal_runs: list[dict] = []
for equal_name, equal_factory in HIGH_CASES:
    for equal_seed in EQUAL_SEEDS:
        equal_run = run_equal_ga(equal_name, equal_factory, equal_seed)
        equal_runs.append(equal_run)
        print(
            f"{equal_name:<14} seed={equal_seed:2d} "
            f"BO={equal_run['bo_best_value']:.9g}, GA={equal_run['ga_best_value']:.9g}, "
            f"shared_init={equal_run['shared_initial_best']:.9g}",
            flush=True,
        )

equal_summary: list[dict] = []
print("同初始样本、同 120 次评估：三个种子的中位数，越小越好")
print(f"{'Case':<15} {'BO median':>14} {'GA median':>14} {'BO gap median':>15} {'GA gap median':>15} {'Winner':>8}")
for equal_name, _ in HIGH_CASES:
    case_equal_runs = [row for row in equal_runs if row["function"] == equal_name]
    bo_equal_median = float(np.median([row["bo_best_value"] for row in case_equal_runs]))
    ga_equal_median = float(np.median([row["ga_best_value"] for row in case_equal_runs]))
    bo_equal_gap = float(np.median([row["bo_gap"] for row in case_equal_runs]))
    ga_equal_gap = float(np.median([row["ga_gap"] for row in case_equal_runs]))
    equal_winner = "BO" if bo_equal_median < ga_equal_median else "GA" if ga_equal_median < bo_equal_median else "Tie"
    equal_summary.append({
        "function": equal_name, "ndim": case_equal_runs[0]["ndim"],
        "evaluations": EQUAL_BUDGET, "known_optimum": case_equal_runs[0]["known_optimum"],
        "bo_median": bo_equal_median, "ga_median": ga_equal_median,
        "bo_gap_median": bo_equal_gap, "ga_gap_median": ga_equal_gap,
        "bo_best": min(row["bo_best_value"] for row in case_equal_runs),
        "ga_best": min(row["ga_best_value"] for row in case_equal_runs),
        "bo_worst": max(row["bo_best_value"] for row in case_equal_runs),
        "ga_worst": max(row["ga_best_value"] for row in case_equal_runs),
        "winner_by_median": equal_winner,
        "ga_paired_wins": sum(row["ga_best_value"] < row["bo_best_value"] for row in case_equal_runs),
    })
    print(f"{equal_name:<15} {bo_equal_median:>14.9g} {ga_equal_median:>14.9g} {bo_equal_gap:>15.9g} {ga_equal_gap:>15.9g} {equal_winner:>8}")
print("\nBO 中位数更好的组数:", sum(row["winner_by_median"] == "BO" for row in equal_summary))
print("GA 中位数更好的组数:", sum(row["winner_by_median"] == "GA" for row in equal_summary))
print("只比较 120 次，不能用前一节 GA 6000 次的结果判定同预算胜负。")

import matplotlib.pyplot as plt

equal_fig, equal_axes = plt.subplots(4, 2, figsize=(13, 15))
for equal_axis, (equal_name, _) in zip(equal_axes.flat, HIGH_CASES):
    case_equal_runs = [row for row in equal_runs if row["function"] == equal_name]
    equal_optimum = case_equal_runs[0]["known_optimum"]
    equal_counts = np.arange(1, EQUAL_BUDGET + 1)
    for method, color in [("bo", "tab:blue"), ("ga", "tab:orange")]:
        equal_gaps = np.asarray([row[f"{method}_best_history"] for row in case_equal_runs]) - equal_optimum
        if method == "ga":
            # GA 同一代的候选解批量产生，不将批内排序视为顺序决策。
            indices = np.arange(EQUAL_POP_SIZE - 1, EQUAL_BUDGET, EQUAL_POP_SIZE)
        else:
            indices = np.arange(EQUAL_BUDGET)
        equal_axis.plot(equal_counts[indices], np.maximum(np.median(equal_gaps, axis=0)[indices], 1e-12), color=color, label=f"{method.upper()} median gap")
        equal_axis.fill_between(
            equal_counts[indices], np.maximum(np.min(equal_gaps, axis=0)[indices], 1e-12),
            np.maximum(np.max(equal_gaps, axis=0)[indices], 1e-12), color=color, alpha=0.15,
        )
    equal_axis.axvline(EQUAL_POP_SIZE, color="gray", linestyle=":")
    equal_axis.set(title=equal_name, xlabel="Function evaluations", ylabel="Best value - known optimum", yscale="log")
    equal_axis.grid(alpha=0.25)
    equal_axis.legend(fontsize=8)
equal_fig.suptitle("BO vs GA: same 120 evaluations and same 20 initial samples, 3 seeds")
equal_fig.tight_layout()
plt.show()

import csv

equal_result_dir = Path("data")
equal_result_dir.mkdir(exist_ok=True)
equal_experiment = {
    "versions": {name: version(name) for name in ["pymoo", "bayesian-optimization", "opfunu", "numpy", "scipy"]},
    "settings": {
        "evaluations_per_run": EQUAL_BUDGET, "seeds": EQUAL_SEEDS,
        "shared_initial_points": EQUAL_POP_SIZE, "ga_pop_size": EQUAL_POP_SIZE,
        "ga_n_offsprings": EQUAL_POP_SIZE, "ga_crossover": "default SBX",
        "ga_mutation": "default PM", "ga_eliminate_duplicates": True,
        "bo_settings": equal_bo_data["settings"],
        "bo_source": "data/bayesian_optimization_highdim_results.json",
        "bounds": "opfunu full default bounds", "comparison": "minimize, median over seeds",
    },
    "summary": equal_summary, "runs": equal_runs,
}
equal_json_path = equal_result_dir / "bo_vs_ga_equal_budget.json"
equal_json_path.write_text(json.dumps(equal_experiment, ensure_ascii=False, indent=2) + "\n")
equal_csv_path = equal_result_dir / "bo_vs_ga_equal_budget.csv"
with equal_csv_path.open("w", newline="", encoding="utf-8") as equal_csv_file:
    equal_writer = csv.DictWriter(equal_csv_file, fieldnames=list(equal_summary[0]))
    equal_writer.writeheader()
    equal_writer.writerows(equal_summary)
print("同预算比较完整记录:", equal_json_path)
print("同预算比较汇总表:", equal_csv_path)

import json
from pathlib import Path
import numpy as np
from IPython.display import Markdown, display
from optimization_benchmark import make_cases, configuration, run_suite

suite_path = Path("data/optimization_suite_results.json")
suite_cases = make_cases()
suite_data: dict = {}
if suite_path.exists():
    suite_data = json.loads(suite_path.read_text())
    if suite_data.get("configuration") != configuration() or [row["name"] for row in suite_data["cases"]] != [case.name for case in suite_cases]:
        suite_data = {}
if not suite_data:
    suite_data = run_suite(workers=4)

suite_summary: list[dict] = suite_data["summary"]
suite_runs: list[dict] = suite_data["runs"]
assert len(suite_summary) == 60 and len(suite_runs) == 180
assert all(row["evaluations"] == 120 for row in suite_runs)
print("函数族:", len({row["family"] for row in suite_summary}))
print("函数与维度组合:", len(suite_summary))
print("配对种子实验:", len(suite_runs))
print("BO / GA / 随机搜索每次评估次数:", suite_data["configuration"]["budget"])
print("详细实现与配置:", Path("optimization_benchmark.py"))

suite_dimensions = sorted({case.ndim for case in suite_cases})
suite_families = list(dict.fromkeys(case.family for case in suite_cases))
coverage_header = "| 函数族 | 类型 | " + " | ".join(f"{d}维" for d in suite_dimensions) + " |"
coverage_lines = [coverage_header, "|---|---|" + "---|" * len(suite_dimensions)]
for family in suite_families:
    family_cases = [case for case in suite_cases if case.family == family]
    supported_dimensions = {case.ndim for case in family_cases}
    marks = ["✓" if d in supported_dimensions else "—" for d in suite_dimensions]
    coverage_lines.append("| " + " | ".join([family, family_cases[0].category, *marks]) + " |")
display(Markdown("### 测试覆盖矩阵\n\n" + "\n".join(coverage_lines)))

suite_result_lines = [
    "| 函数 | 维度 | 已知最优值 | BO 中位数 | GA 中位数 | 随机搜索中位数 | BO/GA 更好 |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for row in suite_summary:
    suite_result_lines.append(
        f"| {row['family']} | {row['ndim']} | {row['known_optimum']:.6g} "
        f"| {row['bo_median']:.6g} | {row['ga_median']:.6g} "
        f"| {row['random_median']:.6g} | {row['winner']} |"
    )
display(Markdown("### 全部 60 组同预算结果\n\n" + "\n".join(suite_result_lines)))
print("BO 胜:", sum(row["winner"] == "BO" for row in suite_summary))
print("GA 胜:", sum(row["winner"] == "GA" for row in suite_summary))
print("近似平局:", sum(row["winner"] == "Tie" for row in suite_summary))

suite_group_lines = [
    "| 分组方式 | 分组 | 组数 | BO 胜 | GA 胜 | 平局 | BO 相对误差 | GA 相对误差 | 随机搜索相对误差 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for group in suite_data["groups"]:
    suite_group_lines.append(
        f"| {group['group_by']} | {group['group']} | {group['cases']} "
        f"| {group['bo_wins']} | {group['ga_wins']} | {group['ties']} "
        f"| {group['bo_median_relative_gap']:.4g} "
        f"| {group['ga_median_relative_gap']:.4g} | {group['random_median_relative_gap']:.4g} |"
    )
display(Markdown("### 按维度和结构类型汇总\n\n" + "\n".join(suite_group_lines)))
print("相对误差为最终误差/初始化误差；分组数字是各组相对误差的中位数。")
print("不同函数尺度、类型及组合数量不同；胜场计数只描述本测试集，不代表普遍优劣。")

import matplotlib.pyplot as plt

suite_overview_fig, suite_overview_axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [2.2, 1]})
suite_heatmap = np.full((len(suite_families), len(suite_dimensions)), np.nan)
for row in suite_summary:
    family_index = suite_families.index(row["family"])
    dimension_index = suite_dimensions.index(row["ndim"])
    gap_floor = 1e-12 * max(1.0, row["initial_gap_median"])
    suite_heatmap[family_index, dimension_index] = np.log10(
        max(row["ga_gap_median"], gap_floor) / max(row["bo_gap_median"], gap_floor)
    )
suite_cmap = plt.get_cmap("RdBu_r").copy()
suite_cmap.set_bad("#eeeeee")
suite_image = suite_overview_axes[0].imshow(suite_heatmap, cmap=suite_cmap, vmin=-3, vmax=3, aspect="auto")
suite_overview_axes[0].set_xticks(range(len(suite_dimensions)), suite_dimensions)
suite_overview_axes[0].set_yticks(range(len(suite_families)), suite_families)
suite_overview_axes[0].set(xlabel="Dimensions", title="Median gap ratio: blue = GA lower, red = BO lower")
for row_index in range(len(suite_families)):
    for column_index in range(len(suite_dimensions)):
        heat_value = suite_heatmap[row_index, column_index]
        if np.isfinite(heat_value):
            suite_overview_axes[0].text(column_index, row_index, f"{heat_value:.1f}", ha="center", va="center", fontsize=8, color="white" if abs(heat_value) > 1.5 else "black")
suite_overview_fig.colorbar(suite_image, ax=suite_overview_axes[0], shrink=0.65, label="log10(GA gap / BO gap), colors clipped to [-3, 3]")
suite_bands = ["low (2-5)", "medium (10-20)", "high (30-50)"]
suite_band_rows = [next(group for group in suite_data["groups"] if group["group_by"] == "dimension_band" and group["group"] == band) for band in suite_bands]
suite_y = np.arange(len(suite_bands))
suite_bo_wins = np.asarray([row["bo_wins"] for row in suite_band_rows])
suite_ga_wins = np.asarray([row["ga_wins"] for row in suite_band_rows])
suite_ties = np.asarray([row["ties"] for row in suite_band_rows])
suite_overview_axes[1].barh(suite_y, suite_bo_wins, label="BO", color="tab:blue")
suite_overview_axes[1].barh(suite_y, suite_ga_wins, left=suite_bo_wins, label="GA", color="tab:orange")
suite_overview_axes[1].barh(suite_y, suite_ties, left=suite_bo_wins + suite_ga_wins, label="Tie", color="gray")
suite_overview_axes[1].set_yticks(suite_y, suite_bands)
suite_overview_axes[1].set(xlabel="Number of cases", title="Wins by dimension band")
suite_overview_axes[1].legend()
suite_overview_axes[1].grid(axis="x", alpha=0.2)
suite_overview_fig.suptitle("60 cases, 21 function families, same 120 evaluations and 20 initial points")
suite_overview_fig.tight_layout()
suite_overview_fig.savefig("data/optimization_suite_overview.png", dpi=150)
plt.show()

suite_selected = [
    "ChungReynolds_2D", "Cigar_10D", "Ackley01_10D", "Ackley01_50D",
    "CEC_F9_10D", "CEC_F9_50D", "CEC_F17_30D", "Corana_4D",
]
suite_curves_fig, suite_curves_axes = plt.subplots(4, 2, figsize=(13, 15))
for curve_axis, selected_case in zip(suite_curves_axes.flat, suite_selected):
    selected_runs = [row for row in suite_runs if row["case"] == selected_case]
    selected_optimum = selected_runs[0]["known_optimum"]
    selected_counts = np.arange(1, 121)
    for method, color in [("bo", "tab:blue"), ("ga", "tab:orange"), ("random", "tab:green")]:
        selected_gaps = np.asarray([row[f"{method}_history"] for row in selected_runs]) - selected_optimum
        selected_indices = np.arange(19, 120, 20) if method == "ga" else np.arange(120)
        curve_axis.plot(selected_counts[selected_indices], np.maximum(np.median(selected_gaps, axis=0)[selected_indices], 1e-12), label=method.upper(), color=color)
        curve_axis.fill_between(
            selected_counts[selected_indices], np.maximum(np.min(selected_gaps, axis=0)[selected_indices], 1e-12),
            np.maximum(np.max(selected_gaps, axis=0)[selected_indices], 1e-12), alpha=0.10, color=color,
        )
    curve_axis.axvline(20, color="gray", linestyle=":")
    curve_axis.set(title=selected_case, xlabel="Function evaluations", ylabel="Best value - known optimum", yscale="log")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(fontsize=8)
suite_curves_fig.suptitle("Selected convergence curves: median and range across 3 seeds")
suite_curves_fig.tight_layout()
suite_curves_fig.savefig("data/optimization_suite_convergence.png", dpi=130)
plt.show()
print("全部结果:", suite_path)
print("每组汇总:", Path("data/optimization_suite_summary.csv"))
print("按维度/类型汇总:", Path("data/optimization_suite_groups.csv"))
print("每个种子的结果:", Path("data/optimization_suite_runs.csv"))