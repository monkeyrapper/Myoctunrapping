# OCT Vibrometry Phase Processing

This repository contains Python scripts for OCT vibrometry phase extraction, no-zero-padding preprocessing, wrapped phase analysis, and real-time Ocean Optics spectrometer monitoring.

## Contents

- `Train_1_3.py`: reference Gram-DCNN / OCT phase processing experiment script.
- `analyze_no_zeropad_oct.py`: batch offline analysis wrapper for no-zero-padding OCT processing.
- `Online_5_5_5.py`: real-time OCT vibrometry UI and TXT spectrum acquisition script.
- `requirements.txt`: Python dependencies.

Large raw spectra, generated figures, model checkpoints, and experiment outputs are intentionally ignored by Git.

## Data Format

The raw spectrum TXT format expected by the offline processing flow is:

```text
Spectrometer Data File
Timestamp: ...
Frame: ...
>>>>>Begin Spectral Data<<<<<
wavelength intensity
wavelength intensity
...
```

File names follow this style:

```text
USB2H031731__0__22-41-31-057.txt
```

which means:

- device prefix: `USB2H031731`
- frame index: `0`
- save timestamp: `22-41-31-057`

## Offline Analysis

Edit `DATASETS` at the top of `analyze_no_zeropad_oct.py`, or pass paths from the command line:

```powershell
python analyze_no_zeropad_oct.py "D:\your_dataset_folder"
```

If the true sampling rate is known, pass it explicitly:

```powershell
python analyze_no_zeropad_oct.py "D:\your_dataset_folder" --sample-rate 1000
```

The no-zero-padding pipeline uses:

- spectrum crop: `200:1848`
- background/DC removal: time mean
- Hann window
- k-space resampling
- FFT without zero padding
- automatic depth peak selection after masking low-depth bins
- wrapped phase extraction from the selected complex depth bin
- sin/cos Savitzky-Golay phase filtering
- automatic unwrap selection

Each output folder contains:

```text
01_dc_check.png
02_depth_profile.png
03_bscan_depth_time.png
04_phase_trace.png
05_displacement_trace.png
06_selected_amplitude.png
oct_phase.csv
oct_phase.npz
processing_summary.json
```

## Real-Time Acquisition

Run:

```powershell
python Online_5_5_5.py
```

The UI:

1. Connects to the first available Ocean Optics spectrometer.
2. Uses no-zero-padding OCT preprocessing.
3. Automatically finds the depth peak.
4. Displays A-scan, wrapped/unwrapped phase, displacement, and spectrum.
5. Can save 5000 raw spectrum frames as TXT files.

Default capture output:

```text
D:\gaungpuyi\captured_txt\capture_YYYYMMDD_HHMMSS
```

## Git Notes

The following are ignored:

- raw datasets: `data/`, `data_new*/`, `capture_*/`
- generated outputs: `runs/`
- model/data artifacts: `*.pt`, `*.npz`, `*.csv`, `*.png`, `*.txt`
- paper PDFs: `*.pdf`

This keeps the GitHub repository focused on code rather than large experimental data.

