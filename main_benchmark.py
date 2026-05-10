from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONDITION_FOLDERS = ["base", "tool", "kg"]
TASK_FAMILIES = ["local", "global", "counterfactual", "conceptual"]
OUTPUT_DIR = ROOT / "benchmark_comparison_outputs"

EXCLUDED_RUN_FILES = {
    "Tools.py",
    "connectivity_graph_data.py",
    "export_utils.py",
    "tool_call_helper.py",
    "base_helper.py",
    "kg_build.py",
    "__init__.py",
}

METRICS = {
    "accuracy": "Accuracy / Final Score",
    "pass_at_5": "PASS@5",
    "pass_5_all_correct": "PASS^5",
    "consistency": "Consistency",
    "ranking_accuracy": "Ranking Accuracy",
    "metric_accuracy": "Metric Accuracy",
}

def clean_api_key(key: str) -> str:
    return key.strip().strip('"').strip("'")

def get_or_prompt_api_key() -> str:
    key = clean_api_key(os.environ.get("ANTHROPIC_API_KEY", ""))
    if key:
        return key
    print("Enter your Anthropic API key once for all benchmark modules:")
    key = clean_api_key(input().strip())
    if not key:
        raise RuntimeError("Missing Anthropic API key.")
    os.environ["ANTHROPIC_API_KEY"] = key
    return key

# Running benchmark modules

def discover_modules(folder: Path) -> List[Path]:
    modules: List[Path] = []
    for task_family in TASK_FAMILIES:
        for pattern in [f"{task_family}_*.py", f"*_{task_family}.py", f"{task_family}.py"]:
            modules.extend(folder.glob(pattern))

    return sorted({
        path
        for path in modules
        if path.is_file()
        and path.name not in EXCLUDED_RUN_FILES
        and not path.name.startswith("__")
        and "(" not in path.name 
    })

def run_module(module_path: Path, shared_env: Optional[Dict[str, str]] = None) -> None:
    env = os.environ.copy()
    if shared_env:
        env.update(shared_env)

    env["OUTPUT_DIR"] = "."

    if clean_api_key(env.get("ANTHROPIC_API_KEY", "")):
        env["SKIP_API_KEY_PROMPT"] = "1"

    print(f"\nRunning: {module_path.relative_to(ROOT)}")
    subprocess.run(
        [sys.executable, module_path.name],
        cwd=str(module_path.parent),
        env=env,
        check=True,
    )

def run_all(condition_folders: Iterable[str], shared_env: Optional[Dict[str, str]] = None) -> None:
    for condition in condition_folders:
        folder = ROOT / condition
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue
        modules = discover_modules(folder)
        if not modules:
            print(f"No benchmark modules found in: {folder}")
            continue
        for module_path in modules:
            run_module(module_path, shared_env=shared_env)

# Summary discovery
def title_tokens(path: Path) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", path.stem.lower()) if t]

def condition_from_path(path: Path) -> Optional[str]:
    parts = [part.lower() for part in path.parts]
    for condition in CONDITION_FOLDERS:
        if condition in parts:
            return condition
    toks = title_tokens(path)
    for condition in CONDITION_FOLDERS:
        if condition in toks:
            return condition
    return None

def task_family_from_title(path: Path) -> Optional[str]:
    toks = title_tokens(path)
    for family in TASK_FAMILIES:
        if family in toks:
            return family
    return None

def is_summary_by_question(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    stem = path.stem.lower()
    return bool(re.search(r"summary[_\- ]*by[_\- ]*question$", stem))

def discover_summary_by_question_files() -> List[Path]:
    candidates: List[Path] = []
    for condition in CONDITION_FOLDERS:
        folder = ROOT / condition
        if not folder.exists():
            continue
        for path in folder.rglob("*.csv"):
            if not is_summary_by_question(path):
                continue
            condition_value = condition_from_path(path)
            family_value = task_family_from_title(path)
            if condition_value is None or family_value is None:
                continue
            candidates.append(path)

    best: Dict[Tuple[str, str], Path] = {}
    for path in candidates:
        condition = condition_from_path(path)
        family = task_family_from_title(path)
        if condition is None or family is None:
            continue
        key = (condition, family)
        if key not in best or len(str(path)) < len(str(best[key])):
            best[key] = path

    return sorted(best.values())
# Metric extraction and rule-based aggregation
def numeric_series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return values
    return None

def first_numeric_mean(df: pd.DataFrame, candidates: List[str]) -> Optional[float]:
    series = numeric_series(df, candidates)
    if series is None:
        return None
    values = series.dropna()
    return float(values.mean()) if not values.empty else None

def row_numeric(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for col in candidates:
        if col in row.index:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None

def exact_accuracy_from_row(row: pd.Series) -> Optional[float]:
    return row_numeric(row, [
        "accuracy",
        "accuracy_over_paraphrases",
        "overall_exact_pass_rate",
        "overall_exact_pass_rate_task",
        "correct",
        "overall_correct",
    ])

def add_rule_based_scores(df: pd.DataFrame, family: str) -> pd.DataFrame:
    out = df.copy()
    scores: List[Optional[float]] = []
    metric_values: List[Optional[float]] = []
    ranking_values: List[Optional[float]] = []
    exact_values: List[Optional[float]] = []

    for _, row in out.iterrows():
        metric_acc = row_numeric(row, ["mean_metric_accuracy", "metric_accuracy"])
        ranking_acc = row_numeric(row, ["mean_ranking_accuracy", "ranking_accuracy"])
        exact_acc = exact_accuracy_from_row(row)

        metric_values.append(metric_acc)
        ranking_values.append(ranking_acc)
        exact_values.append(exact_acc)

        if family == "global":
            if metric_acc is not None and ranking_acc is not None:
                scores.append(0.5 * metric_acc + 0.5 * ranking_acc)
            elif metric_acc is not None:
                scores.append(metric_acc)
            elif ranking_acc is not None:
                scores.append(ranking_acc)
            else:
                scores.append(exact_acc)
        elif family == "counterfactual":
            scores.append(ranking_acc if ranking_acc is not None else exact_acc)
        elif family == "local":
            scores.append(exact_acc)
        elif family == "conceptual":
            scores.append(exact_acc)
        else:
            scores.append(exact_acc)

    out["rule_based_accuracy"] = scores
    out["metric_accuracy_extracted"] = metric_values
    out["ranking_accuracy_extracted"] = ranking_values
    out["exact_accuracy_extracted"] = exact_values
    return out

def analysis_rule_label(family: str) -> str:
    if family == "local":
        return "mean questionwise accuracy per graph/question"
    if family == "global":
        return "mean per-question 0.5*mean_metric_accuracy + 0.5*mean_ranking_accuracy"
    if family == "counterfactual":
        return "mean per-question mean_ranking_accuracy"
    if family == "conceptual":
        return "mean per-task/per-graph question accuracy over paraphrases"
    return "mean rule-based accuracy"

def normalize_summary_file(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    condition = condition_from_path(path)
    family = task_family_from_title(path)
    if condition is None or family is None:
        return None, None

    try:
        raw_df = pd.read_csv(path)
    except Exception as exc:
        print(f"Skipping unreadable CSV {path}: {exc}")
        return None, None

    if raw_df.empty:
        return None, None

    scored_df = add_rule_based_scores(raw_df, family)


    pass_at_5 = first_numeric_mean(scored_df, ["pass_at_5", "pass_at_5_overall", "pass_at_5_task"])
    pass_5_all_correct = first_numeric_mean(scored_df, [
        "pass_5_all_correct",
        "pass_5_all_correct_overall",
        "pass_5_all_correct_task",
    ])
    consistency = first_numeric_mean(scored_df, [
        "consistency",
        "consistency_majority_correctness",
        "consistency_overall_correctness",
        "consistency_answer",
        "consistency_top3_answer",
        "consistency_metric_values",
    ])

    ranking_accuracy = first_numeric_mean(scored_df, ["ranking_accuracy_extracted", "mean_ranking_accuracy", "ranking_accuracy"])
    metric_accuracy = first_numeric_mean(scored_df, ["metric_accuracy_extracted", "mean_metric_accuracy", "metric_accuracy"])

    score_values = pd.to_numeric(scored_df["rule_based_accuracy"], errors="coerce").dropna()
    accuracy = float(score_values.mean()) if not score_values.empty else None

    detail_df = scored_df.copy()

    detail_df["condition"] = condition

    if "task_type" in detail_df.columns:
        detail_df["original_task_type"] = detail_df["task_type"]

    detail_df["task_family"] = family
    detail_df["source_file"] = str(path.relative_to(ROOT))
    detail_df["source_title"] = path.name
    detail_df["analysis_rule"] = analysis_rule_label(family)

    record = {
        "task_type": family,
        "condition": condition,
        "n_question_rows": int(len(scored_df)),
        "accuracy": accuracy,
        "pass_at_5": pass_at_5,
        "pass_5_all_correct": pass_5_all_correct,
        "consistency": consistency,
        "ranking_accuracy": ranking_accuracy,
        "metric_accuracy": metric_accuracy,
        "source_file": str(path.relative_to(ROOT)),
        "source_title": path.name,
        "analysis_rule": analysis_rule_label(family),
    }
    return record, detail_df


def collect_summary_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = discover_summary_by_question_files()
    if not files:
        raise FileNotFoundError(
            "No summary_by_question CSVs found. Expected files named like "
            "local_base_summary_by_question.csv, global_tool_summary_by_question.csv, "
            "counterfactual_kg_summary_by_question.csv, or conceptual_kg_summary_by_question.csv."
        )

    records: List[Dict[str, Any]] = []
    details: List[pd.DataFrame] = []
    for path in files:
        record, detail_df = normalize_summary_file(path)
        if record is not None:
            records.append(record)
        if detail_df is not None:
            details.append(detail_df)

    if not records:
        raise RuntimeError("summary_by_question CSVs were found, but no usable metrics were extracted.")

    summary_df = pd.DataFrame(records)
    summary_df["task_type"] = pd.Categorical(summary_df["task_type"], categories=TASK_FAMILIES, ordered=True)
    summary_df["condition"] = pd.Categorical(summary_df["condition"], categories=CONDITION_FOLDERS, ordered=True)
    summary_df = summary_df.sort_values(["task_type", "condition"]).reset_index(drop=True)

    detail_summary = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary_df, detail_summary

# Export and plotting

def export_dataframe(df: pd.DataFrame, filename: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path

def plot_grouped_bar(df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    if metric not in df.columns or not df[metric].notna().any():
        print(f"Skipping {metric}: no values available.")
        return
    plot_df = df.dropna(subset=[metric]).copy()
    pivot = plot_df.pivot(index="task_type", columns="condition", values=metric).reindex(TASK_FAMILIES)
    existing_conditions = [c for c in CONDITION_FOLDERS if c in pivot.columns]
    pivot = pivot[existing_conditions]
    colors = ["#2f2f2f", "#8a8a8a", "#d0d0d0"][:len(existing_conditions)]
    ax = pivot.plot(kind="bar", figsize=(10, 6), rot=0, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Task Type")
    ax.set_ylabel(METRICS[metric])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Condition", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    available_metrics = [m for m in METRICS if m in df.columns and df[m].notna().any()]
    rows: List[List[float]] = []
    labels: List[str] = []
    for family in TASK_FAMILIES:
        for condition in CONDITION_FOLDERS:
            row = df[(df["task_type"] == family) & (df["condition"] == condition)]
            if row.empty:
                continue
            rec = row.iloc[0]
            rows.append([float(rec[m]) if pd.notna(rec[m]) else 0.0 for m in available_metrics])
            labels.append(f"{family}\n{condition}")
    if not rows:
        print("Skipping heatmap: no values available.")
        return
    heat_df = pd.DataFrame(rows, index=labels, columns=[METRICS[m] for m in available_metrics])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(heat_df.values, aspect="auto", vmin=0, vmax=1, cmap="Greys")
    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index)
    for i in range(heat_df.shape[0]):
        for j in range(heat_df.shape[1]):
            ax.text(j, i, f"{heat_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Benchmark Summary Heatmap", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def make_plots(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for metric, label in METRICS.items():
        if metric in df.columns and df[metric].notna().any():
            path = output_dir / f"{metric}_by_task_type.png"
            plot_grouped_bar(df, metric, f"{label} by Task Type and Condition", path)
            paths.append(path)
    heatmap_path = output_dir / "benchmark_summary_heatmap.png"
    plot_heatmap(df, heatmap_path)
    paths.append(heatmap_path)
    return paths

# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Run and/or analyze benchmark summaries.")
    parser.add_argument("--analysis-only", action="store_true", help="Only analyze existing summary_by_question CSVs.")
    parser.add_argument("--collect-only", action="store_true", help="Alias for --analysis-only.")
    parser.add_argument("--run-only", action="store_true", help="Run benchmark modules but skip analysis.")
    args = parser.parse_args()

    analysis_only = args.analysis_only or args.collect_only
    if not analysis_only:
        api_key = get_or_prompt_api_key()
        run_all(
            CONDITION_FOLDERS,
            shared_env={"ANTHROPIC_API_KEY": api_key, "SKIP_API_KEY_PROMPT": "1"},
        )
        if args.run_only:
            return

    summary_df, detail_df = collect_summary_records()
    summary_path = export_dataframe(summary_df, "benchmark_comparison_summary.csv")
    detail_path = export_dataframe(detail_df, "benchmark_question_level_scores.csv")
    plot_paths = make_plots(summary_df)

    print("\nSaved combined summary:")
    print(f"- {summary_path}")
    print("\nSaved question-level detail:")
    print(f"- {detail_path}")
    print("\nSaved plots:")
    for path in plot_paths:
        print(f"- {path}")
    print("\nCombined summary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
