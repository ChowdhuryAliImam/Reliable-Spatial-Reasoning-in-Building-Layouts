from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def export_path(output_dir, module_name, export_name = None, ext = "csv", *, artifact_name = None):
    
    name = export_name if export_name is not None else artifact_name
    if name is None:
        raise TypeError("export_path() requires export_name or artifact_name")

    safe_module = str(module_name).strip().replace(" ", "_")
    safe_export = str(name).strip().replace(" ", "_")
    return ensure_output_dir(output_dir) / f"{safe_module}_{safe_export}.{ext.lstrip('.')}"

def export_csv(df, output_dir, module_name, export_name = None, *, artifact_name = None):
 
    path = export_path(output_dir, module_name, export_name, "csv", artifact_name=artifact_name)
    df.to_csv(path, index=False)
    return str(path)

def export_tables(tables, output_dir, module_name):
    return {name: export_csv(df, output_dir, module_name, name) for name, df in tables.items()}
def export_csvs(tables, output_dir, module_name):
    return export_tables(tables, output_dir, module_name)

def module_name(file_path):
    return Path(file_path).stem.replace(" ", "_")

def output_dir_for(module_name_value, default = "."):
    return os.getenv("OUTPUT_DIR", default)

def print_saved_outputs(paths):
    print("\nSaved outputs:")
    for path in paths:
        print(f"- {path}")
