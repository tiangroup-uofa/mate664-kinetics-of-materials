#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def contains_import_marimo(py_path: Path) -> bool:
    """Heuristic: only export notebooks that look like marimo apps."""
    try:
        for line in py_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "marimo" in line:
                return True
        return False
    except OSError:
        return False


def export_mode_and_outstem(py_file: Path) -> tuple[str, str]:
    """
    Filename convention:

      foo.py       -> mode="run",  outstem="foo"
      foo.edit.py  -> mode="edit", outstem="foo"
    """
    stem = py_file.stem
    if stem.endswith(".edit"):
        return "edit", stem.removesuffix(".edit")
    return "run", stem


def main() -> None:
    project_root = Path.cwd()
    scripts_dir = project_root / "scripts"

    if not scripts_dir.is_dir():
        raise SystemExit(f"Expected 'scripts/' folder at project root: {scripts_dir}")

    out_dir = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
    if not out_dir:
        raise SystemExit("QUARTO_PROJECT_OUTPUT_DIR is not set")

    quarto_out = Path(out_dir)
    target_dir = quarto_out / "scripts"

    temp_dir_path = Path(tempfile.mkdtemp(prefix="marimo_export_"))

    # Only .py files directly under scripts/
    py_files = sorted(p for p in scripts_dir.iterdir() if p.is_file() and p.suffix == ".py")

    for py_file in py_files:
        if not contains_import_marimo(py_file):
            continue

        mode, outstem = export_mode_and_outstem(py_file)
        out_html = temp_dir_path / f"{outstem}.html"

        if mode == "run":
            cmd = [
                "marimo",
                "export",
                "html-wasm",
                "--no-show-code",
                "--mode",
                "run",
                "-f",
                str(py_file),
                "-o",
                str(out_html),
            ]
        else:  # edit mode
            cmd = [
                "marimo",
                "export",
                "html-wasm",
                "--mode",
                "edit",
                "-f",
                str(py_file),
                "-o",
                str(out_html),
            ]

        print(f"Run {' '.join(cmd)}...")
        subprocess.run(cmd, check=True, cwd=project_root)

    # Replace QUARTO_PROJECT_OUTPUT_DIR/scripts with the freshly exported directory
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(temp_dir_path), str(target_dir))


if __name__ == "__main__":
    main()
