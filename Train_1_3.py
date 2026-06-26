import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# ==========================================
# 1. 深度学习网络定义 (与训练时保持完全一致)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)

class GramDCNN(nn.Module):
    def __init__(self):
        super(GramDCNN, self).__init__()
        self.enc1 = ResidualBlock(1, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(8, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(16, 32)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(32, 64)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(64, 128)
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(128, 64)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(64, 32)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(32, 16)
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(16, 8)
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        return self.final_conv(d1)

# ==========================================
# 2. 编解码与滑动拼接推理模块
# ==========================================
def encode_to_gramian(phase_1d):
    return (phase_1d[:, None] + phase_1d[None, :]) / 2.0

def decode_from_gramian(phase_2d):
    return np.diag(phase_2d).copy()

def predict_long_sequence(model, device, wrapped_phase, window_size=256, overlap=32):
    """
    针对任意长度数据的拼接预测 (复刻论文 Fig 5c)
    """
    model.eval()
    seq_len = len(wrapped_phase)
    stride = window_size - overlap
    
    # 填补序列长度以适应窗口
    pad_len = 0
    if (seq_len - window_size) % stride != 0:
        pad_len = stride - ((seq_len - window_size) % stride)
        wrapped_phase = np.pad(wrapped_phase, (0, pad_len), mode='edge')
    
    num_windows = (len(wrapped_phase) - window_size) // stride + 1
    unwrapped_full = np.zeros_like(wrapped_phase, dtype=np.float64)
    weight_full = np.zeros_like(wrapped_phase, dtype=np.float64)

    with torch.no_grad():
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            segment = wrapped_phase[start:end]
            
            # 推理
            input_matrix = encode_to_gramian(segment)
            input_tensor = torch.FloatTensor(input_matrix).unsqueeze(0).unsqueeze(0).to(device)
            pred_tensor = model(input_tensor)
            pred_matrix = pred_tensor.cpu().squeeze().numpy()
            pred_segment = decode_from_gramian(pred_matrix)
            
            # 解决块与块之间的 2*pi 相位错位问题
            if i > 0:
                # 计算当前块和前序结果在重叠区域的平均相位差
                overlap_prev = unwrapped_full[start : start + overlap]
                overlap_curr = pred_segment[0 : overlap]
                diff = np.mean(overlap_prev - overlap_curr)
                # 寻找最接近的 2*pi 倍数进行对齐补偿
                k_shift = np.round(diff / (2 * np.pi))
                pred_segment += k_shift * 2 * np.pi

            # 重叠相加平滑拼接 (Fade-in / Fade-out)
            fade_in = np.linspace(0, 1, overlap)
            fade_out = 1.0 - fade_in
            
            if i == 0:
                unwrapped_full[start:end] += pred_segment
                weight_full[start:end] += 1
            else:
                # 衰减上一块的尾部，增强当前块的头部
                unwrapped_full[start : start + overlap] = (
                    unwrapped_full[start : start + overlap] * fade_out + 
                    pred_segment[0:overlap] * fade_in
                )
                unwrapped_full[start + overlap : end] += pred_segment[overlap:]
                weight_full[start + overlap : end] += 1

    unwrapped_full = unwrapped_full[:seq_len] # 截断 pad 的部分
    return unwrapped_full

# ==========================================
# 3. OCT 光谱仪数据处理 Pipeline
# ==========================================
def parse_spectrometer_txt(filepath):
    """解析你提供的光谱仪原始文本数据 (已修复 GBK 编码问题)"""
    # 改为 gbk 编码，并忽略无法解析的特殊字符
    with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        
    data_start = 0
    for i, line in enumerate(lines):
        if ">>>>>Begin Spectral Data<<<<<" in line:
            data_start = i + 1
            break
            
    lambdas, intensities = [], []
    for line in lines[data_start:]:
        if line.strip():
            l, i = line.strip().split()
            lambdas.append(float(l))
            intensities.append(float(i))
            
    return np.array(lambdas), np.array(intensities)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

def oct_vibrometry_pipeline(folder_path, target_peak_idx=None, show_depth=False):
    """
    融合正交 S-G 滤波与低频屏蔽的终极 OCT 预处理流水线
    """
    print(">>> 1. 正在加载光谱数据...")
    txt_files = sorted(glob.glob(os.path.join(folder_path, '*.txt')))
    if not txt_files:
        raise ValueError("未找到 .txt 文件！")
        
    spectra = []
    wavelengths = None
    for f in txt_files:
        with open(f, 'r', encoding='gbk', errors='ignore') as file:
            lines = file.readlines()
        data_start = next(i for i, line in enumerate(lines) if ">>>>>Begin Spectral Data<<<<<" in line) + 1
        
        l_temp, i_temp = [], []
        for line in lines[data_start:]:
            if line.strip():
                parts = line.strip().split()
                l_temp.append(float(parts[0]))
                i_temp.append(float(parts[1]))
        spectra.append(i_temp)
        if wavelengths is None:
            wavelengths = np.array(l_temp)
            
    # m_mode 形状: (Time, Pixels)
    m_mode = np.array(spectra)
    
    # --- 1. 光谱裁剪 (去除边缘死区) ---
    crop_start, crop_end = 200, 1848
    wavelengths = wavelengths[crop_start:crop_end]
    m_mode = m_mode[:, crop_start:crop_end]
    
    print(">>> 2. 汉宁窗压制泄露与直流背景扣除...")
    # 【参考代码逻辑 1】：施加汉宁窗
    window = np.hanning(m_mode.shape[1])
    m_mode_windowed = m_mode * window
    
    # 【参考代码逻辑 2】：减去时间维度的平均光谱 (扣除直流项)
    m_mode_windowed = m_mode_windowed - np.mean(m_mode_windowed, axis=0, keepdims=True)
    
    print(">>> 3. lambda -> k 空间重采样 (校正非线性)...")
    k_space = 2 * np.pi / wavelengths
    k_space = k_space[::-1] 
    m_mode_windowed = m_mode_windowed[:, ::-1]
    
    k_linear = np.linspace(k_space.min(), k_space.max(), len(k_space))
    m_mode_k = np.zeros_like(m_mode_windowed)
    for i in range(m_mode_windowed.shape[0]):
        cs = CubicSpline(k_space, m_mode_windowed[i, :])
        m_mode_k[i, :] = cs(k_linear)
        
    print(">>> 4. FFT 提取深度域信息...")
    # 【参考代码逻辑 3】：FFT 提取
    fft_result = np.fft.fft(m_mode_k, axis=1)
    fft_half = fft_result[:, :fft_result.shape[1]//2]
    
    # 【参考代码逻辑 4】：强制屏蔽前 50 个低频像素点
    mean_intensity = np.mean(np.abs(fft_half), axis=0)
    mean_intensity[:50] = 0  # 彻底屏蔽直流干扰峰！
    
    if target_peak_idx is None:
        target_peak_idx = np.argmax(mean_intensity)
        print(f"    自动锁定目标反射面索引: {target_peak_idx}")
    else:
        print(f"    使用手动指定的反射面索引: {target_peak_idx}")
        
    if show_depth:
        plt.figure(figsize=(10, 4))
        plt.plot(mean_intensity)
        plt.axvline(x=target_peak_idx, color='r', linestyle='--', label=f'Locked Peak: {target_peak_idx}')
        plt.title("A-scan Depth Profile (Low-frequencies Masked)")
        plt.xlabel("Depth Index")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

# 【参考代码逻辑 5】：提取原始复数信号
    complex_signal = fft_half[:, target_peak_idx]
    
    # ---------------------------------------------------------
    # 【核心改进点】：复平面静态矢量校正 (I-Q Centering)
    # 减去该深度的复数时间均值，强制将旋转相量拉回绝对坐标原点
    # 这是消除相位“非对称挤压”畸变的决定性一步！
    # ---------------------------------------------------------
    complex_signal_centered = complex_signal - np.mean(complex_signal)
    
    # 提取绝对定心后的包裹相位
    raw_wrapped_phase = np.angle(complex_signal_centered)
    
    print(">>> 5. 执行正交 S-G 滤波 (平滑散斑噪声，保留垂直跳变)...")
    # 【参考代码逻辑 6】：相位域正交滤波
    # 既然已经定心，可以直接用复数的归一化实部和虚部代替 sin 和 cos，物理意义更严谨
    norm_amplitude = np.abs(complex_signal_centered)
    
    # 避免除以零的极小值保护
    norm_amplitude[norm_amplitude == 0] = 1e-10 
    
    sin_p = complex_signal_centered.imag / norm_amplitude
    cos_p = complex_signal_centered.real / norm_amplitude
    
    # 使用 Savitzky-Golay 滤波器 (窗口大小 21, 3阶多项式)
    wl = 21 
    poly = 3
    sin_filtered = savgol_filter(sin_p, window_length=wl, polyorder=poly)
    cos_filtered = savgol_filter(cos_p, window_length=wl, polyorder=poly)
    
    # 【参考代码逻辑 7】：合成干净且绝对对称的包裹相位
    smoothed_wrapped_phase = np.arctan2(sin_filtered, cos_filtered)
    
    return smoothed_wrapped_phase
    


# ==========================================
# 4. 主控程序 (Main)
# ==========================================
if __name__ == "__main__":
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "gram_dcnn_weights.pth" # 你刚才训练保存的模型
    
    # ---------------------------------------------------------
    # 替换为存放你连续帧 .txt 文件的文件夹路径！
    # （为了测试，如果该文件夹只有1个文件，程序也能跑通，但只是1个点）
    # 建议准备 500~1000 个连续采集的 txt 文件进行真实的频率验证
    # ---------------------------------------------------------
    DATA_FOLDER = "./my_oct_data1" 
    
    print("============== 开始加载模型 ==============")
    model = GramDCNN().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("成功加载已训练的 Gram-DCNN 权重！")
    else:
        print("警告: 未找到模型权重文件，使用的是未经训练的随机初始化网络！")
        
    print("\n============== 处理真实 OCT 数据 ==============")
    # 步骤 A: 物理预处理，获取包裹相位时间序列
    # (注意：由于我本地没有你的文件夹，此处代码为了演示能够运行，我临时生成一段模拟序列)
    try:
        wrapped_phase_signal = oct_vibrometry_pipeline(DATA_FOLDER,target_peak_idx=65)
    except Exception as e:
        print(f"读取真实文件失败 ({e})，为展示流程，将自动生成一段 1000 点的连续包裹相位替代...")
        # 【测试用模拟数据】：模拟一个 1000 长度的包含系统噪声的包裹信号
        time_axis = np.linspace(0, 1, 1000)
        true_vib = 15 * np.sin(2 * np.pi * 5 * time_axis) + 5 * np.cos(2 * np.pi * 12 * time_axis)
        noise = np.random.normal(0, 0.8, 1000)
        wrapped_phase_signal = np.angle(np.exp(1j * (true_vib + noise)))
    
    print("\n============== DCNN 深度解包 ==============")
    # 步骤 B: 投入我们训练好的模型进行解包与长序列自动拼接
    unwrapped_signal = predict_long_sequence(model, device, wrapped_phase_signal, window_size=256, overlap=32)
    
    print("\n============== 绘制并保存最终振动恢复曲线 ==============")
    plt.figure(figsize=(14, 6))
    
    # 子图 1：包裹相位
    plt.subplot(2, 1, 1)
    plt.plot(wrapped_phase_signal, color='orange', label='Extracted Wrapped Phase (Noisy)', linewidth=1)
    plt.title("Step 1: Wrapped Phase Extracted from FFT Peak")
    plt.ylabel("Phase (rad)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 子图 2：深度学习解缠后的绝对相位（真实振动信号）
    plt.subplot(2, 1, 2)
    plt.plot(unwrapped_signal, color='red', label='Gram-DCNN Unwrapped Real Phase', linewidth=2)
    plt.title("Step 2: Final Vibration Signal (After DL Unwrapping & Stitching)")
    plt.xlabel("Frame Index (Time: 1 ms / frame)")
    plt.ylabel("Phase (rad)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # ==========================================
    # 新增：自动化图像保存模块
    # ==========================================
    # 1. 定义保存的文件名（可以更改后缀为 .jpg, .pdf, .svg 等）
    save_filename = "OCT_Vibration_Result.png"
    
    # 2. 执行保存
    # dpi=300：指定 300 高分辨率，完全满足学术论文和技术报告的打印出版标准
    # bbox_inches='tight'：极其重要！自动裁剪图像周围多余的白边，防止坐标轴标签被切掉
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    # 3. 打印出绝对路径，方便你在电脑里直接找到它
    print(f" >>> [保存成功] 结果图像已导出至: {os.path.abspath(save_filename)}")
    # ==========================================
    
    # 绘制完成后在屏幕上弹窗显示（必须在 savefig 之后！）
    plt.show()