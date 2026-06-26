# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pyqtgraph as pg
import seabreeze.spectrometers as sb
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.fft import fft
from scipy.signal import savgol_filter


class RealTimeOCTMonitor:
    def __init__(self):
        print("=======================================")
        print("[系统启动] 正在连接 Ocean Optics 光谱仪...")

        self.app = QtWidgets.QApplication([])
        try:
            self.spec = sb.Spectrometer.from_first_available()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                None,
                "光谱仪连接失败",
                f"请检查 USB 连接，并确保 OceanView 已关闭。\n\n{exc}",
            )
            raise

        self.spec.integration_time_micros(1000)
        self.wavelengths = self.spec.wavelengths()
        self.num_pixels = len(self.wavelengths)
        print(f"成功连接 {self.spec.model}, pixels={self.num_pixels}")

        # Match the offline no-zero-padding pipeline used in Train_1_3.py.
        self.crop_start = 200
        self.crop_end = min(1848, self.num_pixels)
        self.proc_wavelengths = self.wavelengths[self.crop_start:self.crop_end]
        self.proc_pixels = len(self.proc_wavelengths)
        self.center_lambda = float(np.mean(self.proc_wavelengths))

        self.k_space = (2.0 * np.pi / self.proc_wavelengths)[::-1]
        self.k_linear = np.linspace(self.k_space.min(), self.k_space.max(), self.proc_pixels)
        self.k_window = np.hanning(self.proc_pixels)

        print("[1/2] 采集背景光谱，请保持光学平台静止 1 秒...")
        QtCore.QThread.msleep(1000)
        self.bg_spectrum = self.spec.intensities()
        print("背景采集完成")
        print("[2/2] 使用不零填充流程自动寻峰: crop=200:1848, mask first 50 bins")
        self.depth_index = self.auto_find_depth_index()
        print(f"自动锁定 depth_index={self.depth_index}")
        print("=======================================")

        self.prev_phase = 0.0
        self.phase_offset = 0.0

        self.history_len = 1000
        self.disp_history_raw = np.zeros(self.history_len)
        self.disp_history_smooth = np.zeros(self.history_len)
        self.phase_history_wrapped = np.zeros(self.history_len)
        self.phase_history_unwrapped = np.zeros(self.history_len)

        self.median_buffer_size = 3
        self.median_buffer = np.zeros(self.median_buffer_size)
        self.alpha_smooth = 0.15
        self.current_smoothed_val = 0.0

        # Causal display filter: filter sin/cos, then atan2. No I/Q recentering.
        self.phase_filter_width = 21
        self.phase_filter_poly = 3
        self.phase_sin_buffer = np.zeros(self.phase_filter_width)
        self.phase_cos_buffer = np.ones(self.phase_filter_width)
        self.phase_filter_count = 0

        self.fft_window_len = 250
        self.fft_window = np.hanning(self.fft_window_len)
        self.fs = 1000.0
        self.freq_axis = np.fft.rfftfreq(self.history_len, 1.0 / self.fs)

        # TXT acquisition settings.
        self.capture_frame_count = 5000
        self.capture_root = Path(r"D:\gaungpuyi\captured_txt")
        self.capture_abort = False
        self.device_prefix = self.get_device_prefix()

        self.build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(10)

    def build_ui(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.win = pg.GraphicsLayoutWidget(show=True, title="SD-OCT 实时测振监测")
        self.win.resize(1000, 850)

        self.control_win = QtWidgets.QWidget()
        self.control_win.setWindowTitle("OCT TXT 数据采集")
        control_layout = QtWidgets.QHBoxLayout(self.control_win)
        self.capture_button = QtWidgets.QPushButton("采集5000帧TXT")
        self.stop_capture_button = QtWidgets.QPushButton("停止采集")
        self.capture_status = QtWidgets.QLabel("未采集")
        self.capture_button.clicked.connect(self.capture_txt_dataset)
        self.stop_capture_button.clicked.connect(self.request_stop_capture)
        control_layout.addWidget(self.capture_button)
        control_layout.addWidget(self.stop_capture_button)
        control_layout.addWidget(self.capture_status)
        self.control_win.show()

        self.plot_ascan = self.win.addPlot(title=f"1. A-scan 深度谱: 不零填充, auto depth_index={self.depth_index}")
        self.plot_ascan.setLabel("bottom", "Depth index")
        self.curve_ascan = self.plot_ascan.plot(pen=pg.mkPen(color=(220, 0, 0), width=1))
        self.plot_ascan.addItem(pg.InfiniteLine(pos=self.depth_index, angle=90, pen="b"))

        self.win.nextRow()

        self.plot_phase = self.win.addPlot(title="2. 相位对比: wrapped 原始相位 / unwrapped 解缠相位")
        self.plot_phase.setLabel("bottom", "Time point (ms)")
        self.plot_phase.setLabel("left", "Phase (rad)")
        self.curve_phase_wrapped = self.plot_phase.plot(
            pen=pg.mkPen(color=(30, 120, 220), width=1),
            name="wrapped",
        )
        self.curve_phase_unwrapped = self.plot_phase.plot(
            pen=pg.mkPen(color=(240, 120, 0), width=2),
            name="unwrapped",
        )

        self.win.nextRow()

        self.plot_disp = self.win.addPlot(title="3. 实时位移: raw / median+EMA")
        self.plot_disp.setLabel("bottom", "Time point (ms)")
        self.plot_disp.setLabel("left", "Displacement (nm)")
        self.curve_disp_raw = self.plot_disp.plot(pen=pg.mkPen(color=(0, 150, 0, 100), width=1))
        self.curve_disp_smooth = self.plot_disp.plot(pen=pg.mkPen(color=(0, 120, 0), width=2))

        self.win.nextRow()

        self.plot_fft = self.win.addPlot(title="4. 实时振动频谱")
        self.plot_fft.setLabel("bottom", "Frequency (Hz)")
        self.plot_fft.setLabel("left", "Relative amplitude")
        self.plot_fft.setXRange(0, 150)
        self.curve_fft = self.plot_fft.plot(pen=pg.mkPen(color=(180, 120, 0), width=2))

    def get_device_prefix(self):
        for name in ("serial_number", "serial", "model"):
            value = getattr(self.spec, name, "")
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = ""
            if value:
                prefix = re.sub(r"[^0-9A-Za-z_-]+", "", str(value))
                if prefix:
                    return prefix
        return "USB2H031731"

    def spectrum_to_ascan(self, raw_spectrum):
        spectrum = raw_spectrum[self.crop_start:self.crop_end]
        background = self.bg_spectrum[self.crop_start:self.crop_end]
        sig_no_dc = spectrum - background
        sig_windowed = sig_no_dc * self.k_window

        # Same direction as Train_1_3.py: reverse wavelength data into increasing k, resample, then FFT.
        sig_k = np.interp(self.k_linear, self.k_space, sig_windowed[::-1])
        return fft(sig_k)

    def auto_find_depth_index(self, frames=10, min_index=50):
        profile = np.zeros(self.proc_pixels // 2, dtype=np.float64)
        for _ in range(frames):
            ascan_complex = self.spectrum_to_ascan(self.spec.intensities())
            profile += np.abs(ascan_complex[: self.proc_pixels // 2])
            QtCore.QThread.msleep(10)

        profile[:min_index] = 0.0
        depth_index = int(np.argmax(profile))
        if depth_index <= 0:
            raise RuntimeError("自动寻峰失败：未找到有效深度峰")
        return depth_index

    def filter_phase_savgol(self, phase):
        self.phase_sin_buffer[:-1] = self.phase_sin_buffer[1:]
        self.phase_cos_buffer[:-1] = self.phase_cos_buffer[1:]
        self.phase_sin_buffer[-1] = np.sin(phase)
        self.phase_cos_buffer[-1] = np.cos(phase)
        self.phase_filter_count += 1

        if self.phase_filter_count < self.phase_filter_width:
            return float(phase)

        sin_filtered = savgol_filter(
            self.phase_sin_buffer,
            window_length=self.phase_filter_width,
            polyorder=self.phase_filter_poly,
            mode="interp",
        )
        cos_filtered = savgol_filter(
            self.phase_cos_buffer,
            window_length=self.phase_filter_width,
            polyorder=self.phase_filter_poly,
            mode="interp",
        )
        return float(np.arctan2(sin_filtered[-1], cos_filtered[-1]))

    def format_spectrum_filename(self, frame_index):
        now = datetime.now()
        stamp = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}-{now.microsecond // 1000:03d}"
        return f"{self.device_prefix}__{frame_index}__{stamp}.txt"

    def save_spectrum_txt(self, output_dir, frame_index, intensities):
        path = output_dir / self.format_spectrum_filename(frame_index)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("Spectrometer Data File\n")
            f.write(f"Timestamp: {datetime.now().isoformat(timespec='milliseconds')}\n")
            f.write(f"Frame: {frame_index}\n")
            f.write(">>>>>Begin Spectral Data<<<<<\n")
            for wavelength, intensity in zip(self.wavelengths, intensities):
                f.write(f"{wavelength:.3f}\t{float(intensity):.6f}\n")
        return path

    def request_stop_capture(self):
        self.capture_abort = True

    def capture_txt_dataset(self):
        self.capture_abort = False
        output_dir = self.capture_root / datetime.now().strftime("capture_%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_button.setEnabled(False)
        self.capture_status.setText(f"采集中: {output_dir}")
        self.timer.stop()

        try:
            for i in range(self.capture_frame_count):
                if self.capture_abort:
                    break
                intensities = self.spec.intensities()
                self.save_spectrum_txt(output_dir, i, intensities)
                if i % 20 == 0:
                    self.capture_status.setText(f"采集中 {i}/{self.capture_frame_count}: {output_dir}")
                    QtWidgets.QApplication.processEvents()
            self.capture_status.setText(f"完成: {output_dir}")
        finally:
            self.capture_button.setEnabled(True)
            self.timer.start(10)

    def update_data(self):
        raw_spectrum = self.spec.intensities()
        ascan_complex = self.spectrum_to_ascan(raw_spectrum)

        current_phase = np.angle(ascan_complex[self.depth_index])
        current_phase = self.filter_phase_savgol(current_phase)
        delta_phi = current_phase - self.prev_phase

        if delta_phi > np.pi:
            self.phase_offset -= 2.0 * np.pi
        elif delta_phi < -np.pi:
            self.phase_offset += 2.0 * np.pi

        self.prev_phase = current_phase
        unwrapped_phase = current_phase + self.phase_offset

        displacement_nm = (self.center_lambda / (4.0 * np.pi)) * unwrapped_phase

        self.median_buffer[:-1] = self.median_buffer[1:]
        self.median_buffer[-1] = displacement_nm
        median_filtered_nm = np.median(self.median_buffer)

        if self.current_smoothed_val == 0.0:
            self.current_smoothed_val = median_filtered_nm
        else:
            self.current_smoothed_val = (
                self.alpha_smooth * median_filtered_nm
                + (1.0 - self.alpha_smooth) * self.current_smoothed_val
            )

        self.disp_history_raw[:-1] = self.disp_history_raw[1:]
        self.disp_history_raw[-1] = displacement_nm
        self.disp_history_smooth[:-1] = self.disp_history_smooth[1:]
        self.disp_history_smooth[-1] = self.current_smoothed_val
        self.phase_history_wrapped[:-1] = self.phase_history_wrapped[1:]
        self.phase_history_wrapped[-1] = current_phase
        self.phase_history_unwrapped[:-1] = self.phase_history_unwrapped[1:]
        self.phase_history_unwrapped[-1] = unwrapped_phase

        disp_display_raw = self.disp_history_raw - np.mean(self.disp_history_raw)
        disp_display_smooth = self.disp_history_smooth - np.mean(self.disp_history_smooth)
        phase_display_unwrapped = self.phase_history_unwrapped - np.mean(self.phase_history_unwrapped)

        recent_data = disp_display_smooth[-self.fft_window_len:]
        windowed_recent = recent_data * self.fft_window
        padded_data = np.pad(windowed_recent, (0, self.history_len - self.fft_window_len), "constant")
        fft_result = np.abs(np.fft.rfft(padded_data))
        fft_result[0:3] = 0

        self.curve_ascan.setData(np.abs(ascan_complex[: self.proc_pixels // 2]))
        self.curve_phase_wrapped.setData(self.phase_history_wrapped)
        self.curve_phase_unwrapped.setData(phase_display_unwrapped)
        self.curve_disp_raw.setData(disp_display_raw)
        self.curve_disp_smooth.setData(disp_display_smooth)
        self.curve_fft.setData(self.freq_axis, fft_result)

    def run(self):
        QtWidgets.QApplication.instance().exec_()
        if hasattr(self, "spec"):
            self.spec.close()


if __name__ == "__main__":
    monitor = RealTimeOCTMonitor()
    monitor.run()
