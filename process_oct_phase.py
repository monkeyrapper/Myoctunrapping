# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def parse_args():
    p = argparse.ArgumentParser(description="No-zero-padding OCT vibrometry phase analysis.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--sort-mode", choices=["auto", "frame", "timestamp", "name"], default="auto")
    p.add_argument("--spectrum-crop", choices=["pixel"], default="pixel")
    p.add_argument("--pixel-start", type=int, default=200)
    p.add_argument("--pixel-end", type=int, default=1848)
    p.add_argument("--spectrum-crop-pad", type=int, default=0)
    p.add_argument("--dc-mode", choices=["time_mean"], default="time_mean")
    p.add_argument("--spectral-filter", choices=["none"], default="none")
    p.add_argument("--window", choices=["hann", "none"], default="hann")
    p.add_argument("--zero-pad-factor", type=int, default=1)
    p.add_argument("--depth-index", type=int, default=-1)
    p.add_argument("--depth-min-index", type=int, default=50)
    p.add_argument("--depth-half-width", type=int, default=0)
    p.add_argument("--baseline-frames", type=int, default=1)
    p.add_argument("--phase-reference", choices=["none"], default="none")
    p.add_argument("--complex-smooth-width", type=int, default=1)
    p.add_argument("--phase-filter", choices=["none", "savgol"], default="savgol")
    p.add_argument("--phase-filter-width", type=int, default=21)
    p.add_argument("--phase-filter-poly", type=int, default=3)
    p.add_argument("--phase-smooth-width", type=int, default=5)
    p.add_argument("--unwrap-mode", choices=["auto", "numpy"], default="auto")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--dcnn-overlap", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-rate", type=float, default=0.0)
    p.add_argument("--refractive-index", type=float, default=1.0)
    p.add_argument("--save-intermediate", action="store_true")
    return p.parse_args()


def frame_index(path: Path) -> int:
    m = re.search(r"__(\d+)__", path.name)
    return int(m.group(1)) if m else 0


def parse_timestamp_seconds(path: Path) -> float:
    m = re.search(r"__(\d+)__(\d{2})-(\d{2})-(\d{2})-(\d{3})", path.name)
    if not m:
        return float("nan")
    hh, mm, ss, ms = map(int, m.groups()[1:])
    return hh * 3600.0 + mm * 60.0 + ss + ms / 1000.0


def choose_file_order(files, sort_mode):
    if sort_mode == "name":
        return sorted(files, key=lambda p: p.name), "name"
    if sort_mode == "timestamp":
        return sorted(files, key=lambda p: (parse_timestamp_seconds(p), p.name)), "timestamp"
    if sort_mode == "frame":
        return sorted(files, key=lambda p: (frame_index(p), p.name)), "frame"
    indices = [frame_index(p) for p in files]
    if len(set(indices)) != len(indices):
        return sorted(files, key=lambda p: (parse_timestamp_seconds(p), p.name)), "timestamp"
    return sorted(files, key=lambda p: (frame_index(p), p.name)), "frame"


def load_spectrum_txt(path: Path):
    wavelengths = []
    intensities = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if "Begin Spectral Data" in text:
                in_data = True
                continue
            if not in_data or not text:
                continue
            parts = re.split(r"[\s,;]+", text)
            if len(parts) < 2:
                continue
            try:
                wavelengths.append(float(parts[0]))
                intensities.append(float(parts[1]))
            except ValueError:
                pass
    if not wavelengths:
        raise ValueError(f"No spectral data found in {path}")
    return np.asarray(wavelengths), np.asarray(intensities)


def load_all(input_dir: Path, pattern: str, sort_mode: str):
    files, actual_sort = choose_file_order(list(input_dir.glob(pattern)), sort_mode)
    if not files:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")
    wl0, sp0 = load_spectrum_txt(files[0])
    spectra = np.empty((len(files), sp0.size), dtype=np.float64)
    spectra[0] = sp0
    for i, path in enumerate(files[1:], start=1):
        wl, sp = load_spectrum_txt(path)
        if wl.size != wl0.size:
            raise ValueError(f"Pixel count mismatch in {path}")
        spectra[i] = sp
    timestamps = np.asarray([parse_timestamp_seconds(p) for p in files], dtype=np.float64)
    if np.all(np.isfinite(timestamps)):
        timestamps -= timestamps[0]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                timestamps[i:] += 24 * 3600
    else:
        timestamps = np.arange(len(files), dtype=np.float64)
    return wl0, spectra, files, timestamps, actual_sort


def moving_average_1d(x, width):
    width = int(width)
    if width <= 1:
        return x.copy()
    if width % 2 == 0:
        width += 1
    pad = width // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), np.ones(width) / width, mode="valid")


def filter_wrapped_phase(phase, mode, width, poly):
    if mode == "none":
        return phase.copy()
    width = int(width)
    if width % 2 == 0:
        width += 1
    width = min(width, phase.size if phase.size % 2 else phase.size - 1)
    width = max(width, poly + 2 + ((poly + 2) % 2 == 0))
    if width <= poly or width < 3:
        return phase.copy()
    s = savgol_filter(np.sin(phase), width, poly, mode="interp")
    c = savgol_filter(np.cos(phase), width, poly, mode="interp")
    return np.arctan2(s, c)


def save_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_all(out, wl, raw, corrected, depth_axis, profile, depth_index, depth_mag, time_s, amp, wrapped, unwrapped, relative, disp, rel_smooth, disp_smooth):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(wl, raw[0], label="raw first")
    ax.plot(wl, corrected[0], label="dc removed first")
    ax.set_xlabel("wavelength / nm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_dc_check.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(depth_axis, profile)
    ax.axvline(depth_axis[depth_index], color="r", ls="--", label=f"selected bin {depth_index}")
    ax.set_xlabel("depth / mm")
    ax.set_ylabel("mean FFT amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_depth_profile.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    im = ax.imshow(20 * np.log10(depth_mag.T / np.max(depth_mag) + 1e-12), aspect="auto", origin="lower", extent=[time_s[0], time_s[-1], depth_axis[0], depth_axis[-1]], cmap="magma")
    ax.set_xlabel("time / s")
    ax.set_ylabel("depth / mm")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out / "03_bscan_depth_time.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(time_s, wrapped, label="wrapped", alpha=0.65)
    ax.plot(time_s, relative, label="unwrapped relative")
    ax.plot(time_s, rel_smooth, label="relative smooth", lw=2)
    ax.set_xlabel("time / s")
    ax.set_ylabel("phase / rad")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "04_phase_trace.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(time_s, disp, label="raw")
    ax.plot(time_s, disp_smooth, label="smooth", lw=2)
    ax.set_xlabel("time / s")
    ax.set_ylabel("relative displacement / nm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "05_displacement_trace.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(time_s, amp)
    ax.set_xlabel("time / s")
    ax.set_ylabel("selected complex amplitude")
    fig.tight_layout()
    fig.savefig(out / "06_selected_amplitude.png")
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl, spectra, files, timestamps, actual_sort = load_all(Path(args.input_dir), args.pattern, args.sort_mode)

    wl_crop = wl[args.pixel_start:args.pixel_end]
    raw = spectra[:, args.pixel_start:args.pixel_end]
    corrected = raw - np.mean(raw, axis=0, keepdims=True)
    window = np.hanning(corrected.shape[1]) if args.window == "hann" else np.ones(corrected.shape[1])

    k = 2 * np.pi / (wl_crop * 1e-9)
    k_space = k[::-1]
    k_linear = np.linspace(k_space.min(), k_space.max(), k_space.size)
    k_spectra = np.empty_like(corrected)
    for i in range(corrected.shape[0]):
        k_spectra[i] = np.interp(k_linear, k_space, (corrected[i] * window)[::-1])

    nfft = k_spectra.shape[1] * max(1, int(args.zero_pad_factor))
    fft_data = np.fft.fft(k_spectra, n=nfft, axis=1)[:, : nfft // 2]
    depth_mag = np.abs(fft_data)
    profile = np.mean(depth_mag, axis=0)
    profile[: args.depth_min_index] = 0
    depth_index = args.depth_index if args.depth_index >= 0 else int(np.argmax(profile))
    complex_trace = fft_data[:, depth_index]
    wrapped = filter_wrapped_phase(np.angle(complex_trace), args.phase_filter, args.phase_filter_width, args.phase_filter_poly)
    large_jumps = int(np.sum(np.abs(np.diff(wrapped)) > 3.0))
    unwrapped = np.unwrap(wrapped)
    baseline = max(1, min(args.baseline_frames, unwrapped.size))
    relative = unwrapped - np.mean(unwrapped[:baseline])
    rel_smooth = moving_average_1d(relative, args.phase_smooth_width)

    center_wavelength_m = float(np.mean(wl_crop) * 1e-9)
    disp = relative * center_wavelength_m / (4 * np.pi * max(args.refractive_index, 1e-12)) * 1e9
    disp_smooth = rel_smooth * center_wavelength_m / (4 * np.pi * max(args.refractive_index, 1e-12)) * 1e9
    amp = np.abs(complex_trace)

    sample_rate = args.sample_rate if args.sample_rate > 0 else (1.0 / np.median(np.diff(timestamps)[np.diff(timestamps) > 0]))
    time_s = np.arange(len(files)) / args.sample_rate if args.sample_rate > 0 else timestamps

    dk = k_linear[1] - k_linear[0]
    depth_axis = np.pi * np.fft.fftfreq(nfft, d=dk)[: nfft // 2] * 1e3

    rows = []
    for i, path in enumerate(files):
        rows.append({
            "frame": i,
            "source_file": path.name,
            "time_s": time_s[i],
            "amplitude": amp[i],
            "phase_wrapped_rad": wrapped[i],
            "phase_unwrapped_rad": unwrapped[i],
            "unwrap_mode": "numpy",
            "phase_relative_rad": relative[i],
            "phase_relative_smooth_rad": rel_smooth[i],
            "displacement_nm": disp[i],
            "displacement_smooth_nm": disp_smooth[i],
        })
    save_csv(out / "oct_phase.csv", rows)
    np.savez_compressed(out / "oct_phase.npz", time_s=time_s, phase_wrapped=wrapped, phase_relative=relative, displacement_nm=disp, amplitude=amp)
    plot_all(out, wl_crop, raw, corrected, depth_axis, profile, depth_index, depth_mag, time_s, amp, wrapped, unwrapped, relative, disp, rel_smooth, disp_smooth)

    summary = {
        "input_dir": str(args.input_dir),
        "pattern": args.pattern,
        "sort_mode": args.sort_mode,
        "actual_sort_mode": actual_sort,
        "frames": len(files),
        "pixels": int(raw.shape[1]),
        "crop_pixel_start": args.pixel_start,
        "crop_pixel_end": args.pixel_end,
        "nfft": int(nfft),
        "sample_rate_hz": float(sample_rate),
        "selected_depth_index": int(depth_index),
        "selected_depth_mm": float(depth_axis[depth_index]),
        "actual_unwrap_mode": "numpy",
        "phase_wrapped_large_jumps": large_jumps,
        "phase_filter": args.phase_filter,
        "phase_filter_width": args.phase_filter_width,
        "phase_filter_poly": args.phase_filter_poly,
        "phase_relative_min_rad": float(np.min(relative)),
        "phase_relative_max_rad": float(np.max(relative)),
        "phase_relative_std_rad": float(np.std(relative)),
        "phase_relative_smooth_std_rad": float(np.std(rel_smooth)),
        "displacement_min_nm": float(np.min(disp)),
        "displacement_max_nm": float(np.max(disp)),
        "displacement_std_nm": float(np.std(disp)),
        "displacement_smooth_std_nm": float(np.std(disp_smooth)),
        "first_file": files[0].name,
        "last_file": files[-1].name,
    }
    with (out / "processing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"frames={len(files)}, pixels={raw.shape[1]}, sample_rate={sample_rate:.3f} Hz")
    print(f"selected depth bin={depth_index}, depth={depth_axis[depth_index]:.6f} mm")
    print(f"relative displacement std={np.std(disp):.6f} nm")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
