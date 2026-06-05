#!/usr/bin/env python3
"""Run main.py sequentially from a JSON hyperparameter sweep.

Example JSON using a grid:

{
  "base_args": {
    "model": "esm2",
    "protein": "SYUA_HUMAN",
    "train_size": 1.0,
    "list_size": 5,
    "peft_type": "lora",
    "device": "cuda"
  },
  "sweep": {
    "lora_r": [8, 16],
    "lora_alpha": [16, 32],
    "learning_rate": [0.0001, 0.00005]
  },
  "output_csv": "sweep_spearmanr.csv"
}

Example JSON using explicit runs:

{
  "base_args": {"model": "esm2", "protein": "SYUA_HUMAN"},
  "runs": [
    {"train_size": 0.8, "list_size": 5, "peft_type": "lora", "lora_r": 16},
    {"train_size": 1.0, "list_size": 1, "peft_type": "ia3"}
  ]
}

Example JSON using table rows and columns:

{
  "base_args": {"model": "esm2", "protein": "YAP1_HUMAN", "list_size": 5},
  "table_runs": {
    "rows": [
      {"label": "Fine-tune full model", "args": {"peft_type": "none"}},
      {"label": "LoRA 16, 16", "args": {"peft_type": "lora", "lora_r": 16, "lora_alpha": 16}}
    ],
    "columns": [
      {"label": "Data subset 1.0", "args": {"train_size": 1.0}},
      {"label": "Data subset 0.8", "args": {"train_size": 0.8}}
    ]
  },
  "table": {"table_csv": "yap1_spearmanr_table.csv"}
}
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = "sweep_spearmanr.csv"
METADATA_PREFIX = "_"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the JSON sweep config.")
    parser.add_argument("--output_csv", help="Override the CSV path from the JSON config.")
    parser.add_argument("--table_csv", help="Override the wide table CSV path from the JSON config.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable to use for main.py. Defaults to this interpreter.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Sweep config must be a JSON object.")
    return config


def expand_runs(config):
    base_args = config.get("base_args", {})
    if "table_runs" in config:
        table_runs = config["table_runs"]
        rows = table_runs.get("rows", [])
        columns = table_runs.get("columns", [])
        if not rows or not columns:
            raise ValueError("'table_runs' must include non-empty 'rows' and 'columns'.")

        runs = []
        for row in rows:
            for column in columns:
                row_args = row.get("args", {})
                column_args = column.get("args", {})
                runs.append({
                    **base_args,
                    **row_args,
                    **column_args,
                    "_table_row": row["label"],
                    "_table_column": column["label"],
                })
        return runs

    if "runs" in config:
        runs = config["runs"]
        if not isinstance(runs, list):
            raise ValueError("'runs' must be a list of argument objects.")
        return [{**base_args, **run_args} for run_args in runs]

    sweep = config.get("sweep", {})
    if not sweep:
        return [base_args]

    keys = list(sweep.keys())
    values = []
    for key in keys:
        candidates = sweep[key]
        if not isinstance(candidates, list):
            candidates = [candidates]
        values.append(candidates)

    runs = []
    for combo in itertools.product(*values):
        runs.append({**base_args, **dict(zip(keys, combo))})
    return runs


def normalize_flag_name(name):
    name = str(name)
    if name.startswith("-"):
        return name
    return "--" + name


def value_to_strings(value):
    if isinstance(value, bool):
        return [] if value else None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return None
    return [str(value)]


def build_command(python_executable, run_args):
    cmd = [python_executable, "main.py"]
    for key, value in run_args.items():
        if str(key).startswith(METADATA_PREFIX):
            continue
        values = value_to_strings(value)
        if values is None:
            continue
        flag = normalize_flag_name(key)
        cmd.append(flag)
        cmd.extend(values)
    return cmd


def run_metadata(run_args):
    return {
        key.lstrip(METADATA_PREFIX): value
        for key, value in run_args.items()
        if str(key).startswith(METADATA_PREFIX)
    }


def existing_metric_files():
    return set(REPO_ROOT.glob("checkpoints/*/*/*/logs.pkl"))


def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output.append(line)
    return process.wait(), "".join(output)


def load_checkpoint_metrics(path):
    try:
        import torch
    except ImportError:
        return {"metric_error": "torch is required to read logs.pkl"}

    try:
        report = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        report = torch.load(path, map_location="cpu")

    row = {"checkpoint_log": str(path.relative_to(REPO_ROOT))}
    try:
        parts = path.relative_to(REPO_ROOT).parts
        row["model"] = parts[1]
        row["dataset"] = parts[2]
        row["checkpoint"] = parts[3]
    except IndexError:
        pass

    baseline = report.get("baseline") if isinstance(report, dict) else None
    if isinstance(baseline, dict):
        row["baseline_spearmanr"] = baseline.get("spearmanr")

    val_spearmanr = report.get("spearmanr") if isinstance(report, dict) else None
    if val_spearmanr is not None:
        val_spearmanr = list(val_spearmanr)
        if val_spearmanr:
            row["best_val_spearmanr"] = max(val_spearmanr)
            row["final_val_spearmanr"] = val_spearmanr[-1]
            row["val_spearmanr_by_epoch"] = ";".join(str(value) for value in val_spearmanr)

    if isinstance(report, dict):
        row["best_epoch"] = report.get("best_epoch")
    return row


def parse_test_spearmanr(stdout):
    current_dataset = None
    in_report = False
    header = None
    results = {}

    for raw_line in stdout.splitlines():
        dataset_match = re.search(r"Current dataset: (.+?)\*+", raw_line)
        if dataset_match:
            current_dataset = dataset_match.group(1).strip()
            continue

        if "Breakdown results" in raw_line:
            in_report = True
            header = None
            continue

        if in_report and "Saving model predictions" in raw_line:
            in_report = False
            header = None
            continue

        if not in_report:
            continue

        tokens = raw_line.split()
        if not tokens:
            continue
        if "spearmanr" in tokens:
            header = tokens
            continue
        if header is None or len(tokens) < 2:
            continue

        group_name = tokens[0]
        if "spearmanr" not in header:
            continue
        value_index = 1 + header.index("spearmanr")
        if value_index >= len(tokens):
            continue
        try:
            value = float(tokens[value_index])
        except ValueError:
            continue

        key_prefix = current_dataset or "test"
        results[f"{key_prefix}_{group_name}_test_spearmanr"] = value
        if group_name == "all_rest":
            results[f"{key_prefix}_test_spearmanr"] = value
    return results


def append_rows(csv_path, rows):
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    existing_rows = []
    fieldnames = []
    if csv_path.exists():
        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    all_rows = existing_rows + rows

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    return csv_path


def first_present(row, candidates):
    for candidate in candidates:
        value = row.get(candidate)
        if value not in (None, ""):
            return value
    return ""


def write_table_csv(table_config, output_csv):
    if not table_config:
        return None

    source_csv = Path(output_csv)
    if not source_csv.is_absolute():
        source_csv = REPO_ROOT / source_csv
    if not source_csv.exists():
        return None

    table_csv = Path(table_config.get("table_csv", "sweep_spearmanr_table.csv"))
    if not table_csv.is_absolute():
        table_csv = REPO_ROOT / table_csv

    row_field = table_config.get("row_field", "table_row")
    column_field = table_config.get("column_field", "table_column")
    metric_fields = table_config.get(
        "metric_fields",
        ["test_spearmanr", "best_val_spearmanr", "final_val_spearmanr", "baseline_spearmanr"],
    )

    with open(source_csv, newline="") as file:
        rows = list(csv.DictReader(file))

    row_order = table_config.get("row_order")
    if not row_order:
        row_order = []
        for row in rows:
            value = row.get(row_field)
            if value and value not in row_order:
                row_order.append(value)

    column_order = table_config.get("column_order")
    if not column_order:
        column_order = []
        for row in rows:
            value = row.get(column_field)
            if value and value not in column_order:
                column_order.append(value)

    values = {}
    for row in rows:
        row_name = row.get(row_field)
        column_name = row.get(column_field)
        if not row_name or not column_name:
            continue
        metric_value = first_present(row, metric_fields)
        if metric_value == "":
            continue
        values[(row_name, column_name)] = metric_value

    table_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(table_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([table_config.get("label_header", "Training type"), *column_order])
        for row_name in row_order:
            writer.writerow([
                row_name,
                *[values.get((row_name, column_name), "") for column_name in column_order],
            ])
    return table_csv


def main():
    args = parse_args()
    config = load_config(args.config)
    runs = expand_runs(config)
    output_csv = args.output_csv or config.get("output_csv", DEFAULT_OUTPUT_CSV)
    table_config = dict(config.get("table") or {})
    if args.table_csv:
        table_config["table_csv"] = args.table_csv

    if not runs:
        raise ValueError("No runs were generated from the sweep config.")

    for index, run_args in enumerate(runs, start=1):
        cmd = build_command(args.python, run_args)
        started_at = datetime.now().isoformat(timespec="seconds")
        print(f"\n=== Sweep run {index}/{len(runs)} ===")
        print("Command:", " ".join(cmd))

        if args.dry_run:
            rows = [{
                "run_id": index,
                "started_at": started_at,
                "status": "dry_run",
                "command": " ".join(cmd),
                "hyperparameters_json": json.dumps(run_args, sort_keys=True),
                **run_metadata(run_args),
            }]
            csv_path = append_rows(output_csv, rows)
            table_path = write_table_csv(table_config, output_csv)
            print(f"Updated {csv_path}")
            if table_path:
                print(f"Updated {table_path}")
            continue

        before_logs = existing_metric_files()
        return_code, stdout = run_command(cmd)
        finished_at = datetime.now().isoformat(timespec="seconds")
        after_logs = existing_metric_files()
        new_logs = sorted(after_logs - before_logs, key=os.path.getmtime)
        test_metrics = parse_test_spearmanr(stdout)

        rows = []
        if new_logs:
            for log_path in new_logs:
                row = load_checkpoint_metrics(log_path)
                row.update(test_metrics)
                row.update({
                    "run_id": index,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": "success" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "command": " ".join(cmd),
                    "hyperparameters_json": json.dumps(run_args, sort_keys=True),
                    **run_metadata(run_args),
                })
                rows.append(row)
        else:
            rows.append({
                "run_id": index,
                "started_at": started_at,
                "finished_at": finished_at,
                "status": "success" if return_code == 0 else "failed",
                "return_code": return_code,
                "command": " ".join(cmd),
                "hyperparameters_json": json.dumps(run_args, sort_keys=True),
                **run_metadata(run_args),
                **test_metrics,
            })

        csv_path = append_rows(output_csv, rows)
        table_path = write_table_csv(table_config, output_csv)
        print(f"Updated {csv_path}")
        if table_path:
            print(f"Updated {table_path}")

        if return_code != 0 and config.get("stop_on_error", True):
            raise SystemExit(return_code)


if __name__ == "__main__":
    main()
