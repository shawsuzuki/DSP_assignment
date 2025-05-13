import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, data = wavfile.read("aggvoice.wav")
data = data.astype(np.float32)

def extract(signal, fs, center_freq, bandwidth):
    N = len(signal)
    t = np.arange(N) / fs

    ### 周波数のダウンシフト
    shifted = signal * np.exp(-2j * np.pi * center_freq * t)

    ### スペクトル取得
    spectrum = np.fft.fft(shifted)
    freqs = np.fft.fftfreq(N, d=1/fs)

    ### FFTによるバンドパスフィルタ
    mask = np.abs(freqs) < (bandwidth / 2)
    spectrum_filtered = spectrum * mask

    ### 逆FFTにより時間領域に変換
    baseband = np.fft.ifft(spectrum_filtered).real
    return baseband

### 信号に変換
sig8k= extract(data, fs, center_freq=8000, bandwidth=4000)
sig16k = extract(data, fs, center_freq=16000, bandwidth=4000)

### 正規化
sig8k /= np.max(np.abs(sig8k))
sig16k /= np.max(np.abs(sig16k))


### 提出用データ
wavfile.write("sig8k.wav", fs, sig8k.astype(np.float32))
wavfile.write("sig16k.wav", fs, sig16k.astype(np.float32))

t = np.arange(len(sig8k)) / fs

plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.plot(t, sig8k)
plt.title("around 8kHz")
plt.subplot(2,1,2)
plt.plot(t, sig16k)
plt.title("around 16kHz")
plt.tight_layout()
plt.savefig("fft_plot.png")
plt.show()
