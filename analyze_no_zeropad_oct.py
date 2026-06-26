# -*- coding: utf-8 -*-
r"""
Offline OCT vibrometry analysis with the no-zero-padding preprocessing pipeline.

Edit DATASETS below, or pass paths from the command line:

    python analyze_no_zeropad_oct.py
    python analyze_no_zeropad_oct.py D:\data\a D:\data\b --sample-rate 1000

Outputs are written under OUTPUT_ROOT. Each dataset produces:
    01_dc_check.png
    02_depth_profile.png
    03_bscan_depth_time.png
    04_phase_trace.png
    05_displacement_trace.png
    06_selected_amplitude.png
    oct_phase.csv
    oct_phase.npz
    processing_summary.json
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


# =========================
# Edit these paths directly
# =========================
DATASETS = [
    r"D:\gaungpuyi\captured_txt\capture_20260625_232634",

]

OUTPUT_ROOT = r"D:\Codex\Programs_reveal_6_23\runs"

# 0 means use timestamps parsed from filenames.
# If 2000 frames are known to be 2 seconds, set SAMPLE_RATE = 1000.
SAMPLE_RATE = 0.0


def safe_name(path: Path) -> str:
    parts = [p for p in path.parts if p not in (path.anchor, "\\", "/")]
    name = "_".join(parts[-3:]) if len(parts) >= 3 else path.name
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def run_one_dataset(input_dir: Path, output_dir: Path, sample_rate: float) -> dict:
    script_dir = Path(__file__).resolve().parent
    process_script = script_dir / "process_oct_phase.py"
    if not process_script.is_file():
        raise FileNotFoundError(f"Cannot find {process_script}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    cmd = [
        sys.executable,
        str(process_script),
        "--input-dir",
        str(input_dir),
        "--out",
        str(output_dir),
        "--sort-mode",
        "auto",
        "--spectrum-crop",
        "pixel",
        "--pixel-start",
        "200",
        "--pixel-end",
        "1848",
        "--spectrum-crop-pad",
        "0",
        "--dc-mode",
        "time_mean",
        "--spectral-filter",
        "none",
        "--window",
        "hann",
        "--zero-pad-factor",
        "1",
        "--depth-min-index",
        "50",
        "--depth-half-width",
        "0",
        "--baseline-frames",
        "1",
        "--phase-reference",
        "none",
        "--complex-smooth-width",
        "1",
        "--phase-filter",
        "savgol",
        "--phase-filter-width",
        "21",
        "--phase-filter-poly",
        "3",
        "--phase-smooth-width",
        "5",
        "--unwrap-mode",
        "auto",
        "--checkpoint",
        str(script_dir / "runs" / "best.pt"),
        "--dcnn-overlap",
        "32",
        "--device",
        "cuda",
    ]
    if sample_rate and sample_rate > 0:
        cmd.extend(["--sample-rate", str(sample_rate)])

    print("\n=== Processing ===")
    print(f"input:  {input_dir}")
    print(f"output: {output_dir}")
    subprocess.run(cmd, check=True)

    summary_path = output_dir / "processing_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary_csv(rows, output_root: Path) -> None:
    if not rows:
        return
    fields = [
        "name",
        "input_dir",
        "frames",
        "sample_rate_hz",
        "selected_depth_index",
        "selected_depth_mm",
        "actual_unwrap_mode",
        "phase_wrapped_large_jumps",
        "phase_relative_std_rad",
        "displacement_std_nm",
        "displacement_min_nm",
        "displacement_max_nm",
        "out_dir",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "no_zeropad_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-zero-padding OCT vibrometry analysis.")
    parser.add_argument("input_dirs", nargs="*", help="Dataset directories. If empty, DATASETS in this file are used.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--sample-rate", type=float, default=SAMPLE_RATE, help="0 uses filename timestamps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(p) for p in (args.input_dirs or DATASETS)]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for input_dir in input_dirs:
        out_dir = output_root / safe_name(input_dir)
        summary = run_one_dataset(input_dir, out_dir, args.sample_rate)
        summary["name"] = input_dir.name
        summary["out_dir"] = str(out_dir)
        rows.append(summary)

    write_summary_csv(rows, output_root)

    print("\n=== Done ===")
    print(f"summary: {output_root / 'no_zeropad_summary.csv'}")
    for row in rows:
        print(
            f"{row['name']}: depth={row['selected_depth_index']}, "
            f"unwrap={row['actual_unwrap_mode']}, "
            f"disp_std={row['displacement_std_nm']:.3f} nm, "
            f"out={row['out_dir']}"
        )


if __name__ == "__main__":
    main()
