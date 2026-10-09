"""Summarize the optimization_suite_cache into CSV tables and convergence charts.

Consumes the per-case/seed JSON caches written by
``optimization_benchmark.run_suite`` and produces, under ``data/``:

  optimization_suite_report.csv                per-case BO/GA/Random comparison
  optimization_suite_convergence.csv           best-so-far curves by group/method
  optimization_suite_convergence_sampled.csv   same curves at every 10th evaluation
  optimization_suite_winrate.csv               pointwise BO-vs-GA win/loss counts
  optimization_suite_convergence.png           overall + dimension-band panels
  optimization_suite_convergence_categories.png  one panel per function category
  optimization_suite_winrate.png               BO share of pointwise BO/GA wins

Run with the project Python:  .venv/bin/python optimization_suite_report.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("bo", "ga", "random")
METHOD_LABELS = {"bo": "BO (UCB, kappa=2.576)", "ga": "GA (pymoo)", "random": "Random search"}
METHOD_COLORS = {"bo": "#1f77b4", "ga": "#d62728", "random": "#7f7f7f"}

# opfunu 1.0.1 ships a defective modified-Schwefel implementation behind F10/F17,
# so those families are excluded from the primary comparison.
EXCLUDED_FAMILIES = {"CEC_F10", "CEC_F17"}
BAND_ORDER = ["low (2-5)", "medium (10-20)", "high (30-50)"]


def load_runs(cache_dir: Path) -> list[dict]:
    """Load every cached run, dropping the excluded defective-function families."""
    runs = [json.loads(path.read_text()) for path in sorted(cache_dir.glob("*.json"))]
    return [run for run in runs if run["family"] not in EXCLUDED_FAMILIES]


def relative_curve(run: dict, method: str) -> np.ndarray:
    """Best-so-far curve normalized to [0, 1]: 1 = initial gap, 0 = known optimum."""
    history = np.asarray(run[f"{method}_history"], dtype=float)
    span = run["initial_best"] - run["known_optimum"]
    if span <= 0:
        return np.zeros_like(history)
    return np.clip((history - run["known_optimum"]) / span, 0.0, None)


def _run_relative_gap(run: dict, method: str) -> float:
    span = run["initial_best"] - run["known_optimum"]
    return max(0.0, run[f"{method}_gap"]) / max(1e-12, span)


def build_case_rows(runs: list[dict]) -> list[dict]:
    """Collapse the three seeds of each case into one median-based row."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["family"], run["ndim"])].append(run)

    rows: list[dict] = []
    for (family, ndim), case_runs in grouped.items():
        case_runs = sorted(case_runs, key=lambda run: run["seed"])
        optimum = case_runs[0]["known_optimum"]
        initial_gap = st.median(run["initial_best"] - optimum for run in case_runs)
        row: dict = {
            "case": f"{family}_{ndim}D",
            "family": family,
            "ndim": ndim,
            "category": case_runs[0]["category"],
            "dimension_band": case_runs[0]["dimension_band"],
            "known_optimum": optimum,
            "initial_gap_median": initial_gap,
        }
        for method in METHODS:
            best = st.median(run[f"{method}_best"] for run in case_runs)
            row[f"{method}_best_median"] = best
            row[f"{method}_gap_median"] = best - optimum
            row[f"{method}_relative_gap"] = st.median(_run_relative_gap(run, method) for run in case_runs)
            row[f"{method}_seconds_median"] = st.median(run[f"{method}_seconds"] for run in case_runs)
        tolerance = 1e-10 * max(1.0, abs(initial_gap))
        if abs(row["bo_best_median"] - row["ga_best_median"]) <= tolerance:
            row["winner"] = "Tie"
        else:
            row["winner"] = "BO" if row["bo_best_median"] < row["ga_best_median"] else "GA"
        row["bo_paired_wins"] = sum(run["bo_best"] < run["ga_best"] for run in case_runs)
        row["ga_paired_wins"] = sum(run["ga_best"] < run["bo_best"] for run in case_runs)
        rows.append(row)
    rows.sort(key=lambda row: (row["family"], row["ndim"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _curve_stats(case_runs: list[dict], method: str) -> dict[str, np.ndarray]:
    curves = np.vstack([relative_curve(run, method) for run in case_runs])
    return {
        "median": np.median(curves, axis=0),
        "mean": curves.mean(axis=0),
        "p25": np.percentile(curves, 25, axis=0),
        "p75": np.percentile(curves, 75, axis=0),
    }


def build_convergence_rows(
    runs: list[dict],
    scopes: Iterable[tuple[str, str, Callable[[dict], bool]]],
) -> list[dict]:
    """Median best-so-far curve per method for each named scope of runs."""
    rows: list[dict] = []
    for group_type, group, predicate in scopes:
        selected = [run for run in runs if predicate(run)]
        if not selected:
            continue
        for method in METHODS:
            stats = _curve_stats(selected, method)
            for index in range(len(stats["median"])):
                rows.append(
                    {
                        "group_type": group_type,
                        "group": group,
                        "method": method,
                        "evaluation": index + 1,
                        "median_relative_gap": float(stats["median"][index]),
                        "mean_relative_gap": float(stats["mean"][index]),
                        "p25_relative_gap": float(stats["p25"][index]),
                        "p75_relative_gap": float(stats["p75"][index]),
                    }
                )
    return rows


def _scopes(runs: list[dict]) -> list[tuple[str, str, Callable[[dict], bool]]]:
    scopes: list[tuple[str, str, Callable[[dict], bool]]] = [("overall", "all cases", lambda run: True)]
    for band in BAND_ORDER:
        scopes.append(("dimension_band", band, lambda run, band=band: run["dimension_band"] == band))
    for category in sorted({run["category"] for run in runs}):
        scopes.append(("category", category, lambda run, category=category: run["category"] == category))
    for family in sorted({run["family"] for run in runs}):
        scopes.append(("family", family, lambda run, family=family: run["family"] == family))
    return scopes


def build_winrate_rows(
    runs: list[dict],
    scopes: Iterable[tuple[str, str, Callable[[dict], bool]]],
) -> list[dict]:
    """Pointwise BO-vs-GA comparison at every evaluation index.

    For each evaluation t the best-so-far gap is compared per run, so the win
    share answers "how often is BO ahead of GA by this point in the budget".
    """
    rows: list[dict] = []
    for group_type, group, predicate in scopes:
        selected = [run for run in runs if predicate(run)]
        if not selected:
            continue
        bo = np.vstack([relative_curve(run, "bo") for run in selected])
        ga = np.vstack([relative_curve(run, "ga") for run in selected])
        for index in range(bo.shape[1]):
            bo_column, ga_column = bo[:, index], ga[:, index]
            tolerance = 1e-9 * np.maximum(1.0, np.maximum(np.abs(bo_column), np.abs(ga_column)))
            bo_wins = int(np.sum(bo_column < ga_column - tolerance))
            ga_wins = int(np.sum(ga_column < bo_column - tolerance))
            decided = bo_wins + ga_wins
            rows.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "evaluation": index + 1,
                    "n_runs": len(selected),
                    "bo_wins": bo_wins,
                    "ga_wins": ga_wins,
                    "ties": len(selected) - decided,
                    "bo_win_share": bo_wins / decided if decided else 0.5,
                    "median_bo_relative_gap": float(np.median(bo_column)),
                    "median_ga_relative_gap": float(np.median(ga_column)),
                }
            )
    return rows


def _draw_panel(ax, stats: dict[str, dict[str, np.ndarray]], title: str, show_band: bool = True) -> None:
    for method in METHODS:
        median = stats[method]["median"]
        x = np.arange(1, len(median) + 1)
        ax.plot(x, median, color=METHOD_COLORS[method], linewidth=2.0, label=METHOD_LABELS[method])
        ax.fill_between(
            x, stats[method]["p25"], stats[method]["p75"], color=METHOD_COLORS[method], alpha=0.12, linewidth=0
        )
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("函数评估次数")
    ax.set_ylabel("相对 gap（对数）\n1 = 初始差距, 0 = 最优")
    ax.grid(True, which="both", alpha=0.25)
    if show_band:
        ax.axvline(20, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate("共享初始点\n(20)", xy=(20, 0.02), xytext=(26, 0.02), fontsize=8, color="#444444")


def _configure_fonts() -> None:
    from matplotlib import font_manager as fm

    for candidate in ["/Users/junwei/Library/Fonts/NotoSansCJK-wght-400-900.ttf.ttc"]:
        path = Path(candidate)
        if path.exists():
            fm.fontManager.addfont(str(path))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_convergence(cache_dir: Path, output_dir: Path, runs: list[dict]) -> None:
    _configure_fonts()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_specs = [("overall", "all cases", "全部用例（60 个，中位数与四分位带）")]
    panel_specs += [("dimension_band", band, f"维度档：{band}") for band in BAND_ORDER]
    for ax, (group_type, group, title) in zip(axes.ravel(), panel_specs):
        selected = [run for run in runs if group_type == "overall" or run[group_type] == group]
        stats = {method: _curve_stats(selected, method) for method in METHODS}
        _draw_panel(ax, stats, title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("BO vs GA vs Random —— 等预算收敛趋势（120 次评估）", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = output_dir / "optimization_suite_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    categories = sorted({run["category"] for run in runs})
    columns = 3
    rows = (len(categories) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.4 * rows), squeeze=False)
    for ax, category in zip(axes.ravel(), categories):
        selected = [run for run in runs if run["category"] == category]
        stats = {method: _curve_stats(selected, method) for method in METHODS}
        _draw_panel(ax, stats, f"{category}（{len({(r['family'], r['ndim']) for r in selected})} 用例）", show_band=False)
    for ax in axes.ravel()[len(categories):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("各函数类型的收敛趋势", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = output_dir / "optimization_suite_convergence_categories.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


def plot_winrate(winrate_rows: list[dict], output_dir: Path) -> None:
    _configure_fonts()
    overall = [row for row in winrate_rows if row["group_type"] == "overall"]
    bands = {band: [row for row in winrate_rows if row["group_type"] == "dimension_band" and row["group"] == band] for band in BAND_ORDER}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = [row["evaluation"] for row in overall]
    ax.plot(x, [row["bo_win_share"] for row in overall], color="#111111", linewidth=2.6, label="全部用例")
    band_colors = {"low (2-5)": "#2ca02c", "medium (10-20)": "#ff7f0e", "high (30-50)": "#9467bd"}
    for band in BAND_ORDER:
        rows = bands[band]
        ax.plot(
            [row["evaluation"] for row in rows],
            [row["bo_win_share"] for row in rows],
            color=band_colors[band],
            linewidth=1.8,
            label=f"维度档：{band}",
        )
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(20, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.annotate("共享初始点结束 (20)", xy=(20, 0.5), xytext=(24, 0.52), fontsize=8, color="#444444")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("函数评估次数")
    ax.set_ylabel("BO 逐点胜率\n(0.5 = 与 GA 打平)")
    ax.set_title("BO vs GA —— 逐点胜负随时间变化（跨种子配对）", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    path = output_dir / "optimization_suite_winrate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e4 or magnitude < 1e-3:
        return f"{value:.2e}"
    if magnitude >= 100:
        return f"{value:.3g}"
    return f"{value:.4g}"


def _group_stats(case_rows: list[dict], key: str) -> list[dict]:
    groups = []
    for label in sorted({row[key] for row in case_rows}):
        rows = [row for row in case_rows if row[key] == label]
        groups.append(
            {
                "label": label,
                "cases": len(rows),
                "bo_wins": sum(row["winner"] == "BO" for row in rows),
                "ga_wins": sum(row["winner"] == "GA" for row in rows),
                **{
                    f"{method}_relative_gap": st.median(row[f"{method}_relative_gap"] for row in rows)
                    for method in METHODS
                },
            }
        )
    return groups


def write_markdown(case_rows: list[dict], winrate_rows: list[dict], path: Path) -> None:
    total = len(case_rows)
    bo_wins = sum(row["winner"] == "BO" for row in case_rows)
    ga_wins = sum(row["winner"] == "GA" for row in case_rows)
    paired_bo = sum(row["bo_paired_wins"] for row in case_rows)
    paired_ga = sum(row["ga_paired_wins"] for row in case_rows)

    lines = [
        "# BO vs GA vs Random —— 等预算对比汇总",
        "",
        f"有效用例 {total} 个（{len({r['family'] for r in case_rows})} 个函数族，种子 0/1/42）；"
        "每个用例共 120 次函数评估，前 20 点为三方共享初始点，取三次种子中位数。"
        "已排除 CEC_F10/F17（opfunu 1.0.1 的 modified Schwefel 实现缺陷）。",
        "",
        "## 1. 总体结论",
        "",
        f"- 按种子中位数判定：**BO 胜 {bo_wins} / GA 胜 {ga_wins} / 平 {total - bo_wins - ga_wins}**（BO 占 {bo_wins / total:.0%}）",
        f"- 逐种子配对（共 {3 * total} 对）：BO 更优 {paired_bo} 次，GA 更优 {paired_ga} 次",
        f"- 相对 gap 中位数（越小越好）：BO {st.median(r['bo_relative_gap'] for r in case_rows):.3f}，"
        f"GA {st.median(r['ga_relative_gap'] for r in case_rows):.3f}，"
        f"Random {st.median(r['random_relative_gap'] for r in case_rows):.3f}",
        f"- 单用例耗时中位数：BO {st.median(r['bo_seconds_median'] for r in case_rows):.2f}s，"
        f"GA {st.median(r['ga_seconds_median'] for r in case_rows):.4f}s，"
        f"Random {st.median(r['random_seconds_median'] for r in case_rows):.4f}s",
        "",
        "## 2. 收敛趋势",
        "",
        "纵轴为相对 gap（对数）：1 = 初始差距，0 = 已知最优；曲线为跨种子/用例的中位数，阴影为四分位带。",
        "",
        "![整体与维度档收敛](optimization_suite_convergence.png)",
        "",
        "![各函数类型收敛](optimization_suite_convergence_categories.png)",
        "",
        "## 3. 逐点胜负（BO vs GA）",
        "",
        "在每一次评估处，按三次种子配对比较 BO 与 GA 的 best-so-far，胜率 0.5 表示打平。",
        "",
        "![逐点胜率](optimization_suite_winrate.png)",
        "",
    ]

    overall = {row["evaluation"]: row for row in winrate_rows if row["group_type"] == "overall"}
    checkpoints = [20, 40, 60, 80, 100, 120]
    lines += [
        "| 评估次数 | BO胜点 | GA胜点 | 平 | BO胜率 |",
        "|---|---|---|---|---|",
    ]
    for evaluation in checkpoints:
        row = overall[evaluation]
        lines.append(
            f"| {evaluation} | {row['bo_wins']} | {row['ga_wins']} | {row['ties']} | {row['bo_win_share']:.2f} |"
        )
    final = overall[120]
    lines += [
        "",
        f"到 120 次评估时，BO 在 {final['bo_wins']} / {final['bo_wins'] + final['ga_wins']} 个已分出的配对上领先"
        f"（胜率 {final['bo_win_share']:.2f}）；20 次评估（共享初始点结束）时两者完全相同。",
        "",
        "## 4. 按维度档",
        "",
        "| 维度档 | 用例 | BO胜 | GA胜 | BO相对gap | GA相对gap | 随机相对gap |",
        "|---|---|---|---|---|---|---|",
    ]
    band_rows = _group_stats(case_rows, "dimension_band")
    band_order = {label: index for index, label in enumerate(BAND_ORDER)}
    for row in sorted(band_rows, key=lambda item: band_order.get(item["label"], 9)):
        lines.append(
            f"| {row['label']} | {row['cases']} | {row['bo_wins']} | {row['ga_wins']} | "
            f"{row['bo_relative_gap']:.3f} | {row['ga_relative_gap']:.3f} | {row['random_relative_gap']:.3f} |"
        )

    for title, key in [("按函数类型", "category"), ("按函数族", "family")]:
        lines += ["", f"## 5. {title}" if key == "category" else f"## 6. {title}", "",
                  "| 分组 | 用例 | BO胜 | GA胜 | BO相对gap | GA相对gap | 随机相对gap |",
                  "|---|---|---|---|---|---|---|"]
        for row in _group_stats(case_rows, key):
            lines.append(
                f"| {row['label']} | {row['cases']} | {row['bo_wins']} | {row['ga_wins']} | "
                f"{row['bo_relative_gap']:.3f} | {row['ga_relative_gap']:.3f} | {row['random_relative_gap']:.3f} |"
            )

    lines += ["", "## 7. 逐用例明细", "",
              "| 用例 | 维 | 类型 | 已知最优 | 初始gap | BO最优 | GA最优 | 随机最优 | BO相对gap | GA相对gap | 随机相对gap | 胜方 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in case_rows:
        lines.append(
            f"| {row['case']} | {row['ndim']} | {row['category']} | {_fmt(row['known_optimum'])} | "
            f"{_fmt(row['initial_gap_median'])} | {_fmt(row['bo_best_median'])} | {_fmt(row['ga_best_median'])} | "
            f"{_fmt(row['random_best_median'])} | {row['bo_relative_gap']:.3f} | {row['ga_relative_gap']:.3f} | "
            f"{row['random_relative_gap']:.3f} | {row['winner']} |"
        )
    lines += [
        "",
        "## 附：文件",
        "",
        "- `optimization_suite_report.csv`：逐用例明细（本文件第 7 节的机器可读版本）",
        "- `optimization_suite_convergence.csv`：各分组、各方法的 best-so-far 曲线（median/mean/p25/p75）",
        "- `optimization_suite_convergence_sampled.csv`：同上，仅保留第 10/20/…/120 次评估的采样点",
        "- `optimization_suite_winrate.csv`：各分组的逐点 BO-vs-GA 胜负计数与胜率",
        "- `optimization_suite_convergence.png`：整体 + 维度档收敛图",
        "- `optimization_suite_convergence_categories.png`：各函数类型收敛图",
        "- `optimization_suite_winrate.png`：BO 逐点胜率曲线",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main(cache_dir: str = "data/optimization_suite_cache", output_dir: str = "data") -> None:
    cache_path = Path(cache_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(cache_path)
    print(f"Loaded {len(runs)} runs, {len({(r['family'], r['ndim']) for r in runs})} cases")

    case_rows = build_case_rows(runs)
    write_csv(out / "optimization_suite_report.csv", case_rows)
    print(f"Saved {out / 'optimization_suite_report.csv'} ({len(case_rows)} rows)")

    scopes = _scopes(runs)
    convergence_rows = build_convergence_rows(runs, scopes)
    write_csv(out / "optimization_suite_convergence.csv", convergence_rows)
    print(f"Saved {out / 'optimization_suite_convergence.csv'} ({len(convergence_rows)} rows)")

    sampled_rows = [row for row in convergence_rows if row["evaluation"] % 10 == 0]
    write_csv(out / "optimization_suite_convergence_sampled.csv", sampled_rows)
    print(f"Saved {out / 'optimization_suite_convergence_sampled.csv'} ({len(sampled_rows)} rows)")

    winrate_rows = build_winrate_rows(runs, scopes)
    write_csv(out / "optimization_suite_winrate.csv", winrate_rows)
    print(f"Saved {out / 'optimization_suite_winrate.csv'} ({len(winrate_rows)} rows)")

    plot_convergence(cache_path, out, runs)
    plot_winrate(winrate_rows, out)
    write_markdown(case_rows, winrate_rows, out / "optimization_suite_report.md")


if __name__ == "__main__":
    main()
