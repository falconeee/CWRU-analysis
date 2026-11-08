"""Helpers for CWRU_bearing_signals.ipynb. Public Case Western Reserve University (CWRU) data."""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.io import loadmat

FS = 12_000  # Hz, drive-end channel
URL = "https://engineering.case.edu/sites/default/files/{}.mat"
SHAFT_HZ = {0: 1797 / 60, 1: 1772 / 60, 2: 1750 / 60, 3: 1730 / 60}  # nominal speed per load (HP)

# SKF 6205-2RS JEM, drive end: ball count, ball diameter, pitch diameter (in), 0 deg contact angle
N_BALLS, D_BALL, D_PITCH = 9, 0.3126, 1.537

# Drive-end channel, faults seeded in the drive-end bearing, 12 kHz. Lists follow loads 0, 1, 2, 3 HP.
FILES = {
    ("Normal", 0.0): [97, 98, 99, 100],
    ("Inner race", 0.007): [105, 106, 107, 108],
    ("Inner race", 0.014): [169, 170, 171, 172],
    ("Inner race", 0.021): [209, 210, 211, 212],
    ("Ball", 0.007): [118, 119, 120, 121],
    ("Ball", 0.014): [185, 186, 187, 188],
    ("Ball", 0.021): [222, 223, 224, 225],
    ("Outer race", 0.007): [130, 131, 132, 133],
    ("Outer race", 0.014): [197, 198, 199, 200],
    ("Outer race", 0.021): [234, 235, 236, 237],
}
CLASSES = ["Normal", "Inner race", "Outer race", "Ball"]


def fault_orders():
    """Characteristic fault frequencies as multiples of shaft speed, from bearing geometry."""
    r = D_BALL / D_PITCH
    return {
        "BPFO": N_BALLS / 2 * (1 - r),
        "BPFI": N_BALLS / 2 * (1 + r),
        "BSF": (1 / (2 * r)) * (1 - r**2),
        "FTF": 0.5 * (1 - r),
    }


def download(folder="data/cwru"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for numbers in FILES.values():
        for n in numbers:
            target = folder / f"{n}.mat"
            if not target.exists():
                partial = target.with_suffix(".part")
                subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-all-errors", "-m", "180",
                                "-o", str(partial), URL.format(n)], check=True)
                partial.rename(target)  # becomes .mat only once complete
    return folder


def read(n, folder="data/cwru"):
    m = loadmat(Path(folder) / f"{n}.mat")
    keys = [k for k in m if k.endswith("DE_time")]
    # Some files carry more than one recording; prefer the one named after the file.
    preferred = [k for k in keys if k.startswith(f"X{n:03d}")]
    return m[(preferred or keys)[0]].ravel().astype(float)


def rpm(n, folder="data/cwru"):
    """Measured shaft speed stored in the file (XnnnRPM), or None when the file has none."""
    m = loadmat(Path(folder) / f"{n}.mat")
    keys = [k for k in m if k == f"X{n:03d}RPM"]
    return float(m[keys[0]].ravel()[0]) if keys else None


def shaft_hz(n, load, folder="data/cwru"):
    """Shaft frequency in Hz: measured speed when the file has it, nominal speed for the load otherwise."""
    r = rpm(n, folder)
    return r / 60 if r else SHAFT_HZ[load]


def windows(folder="data/cwru", seconds=1.0):
    """Non-overlapping windows, as in the benchmark. Returns signals and metadata."""
    size = int(seconds * FS)
    signals, meta = [], []
    for (label, damage), numbers in FILES.items():
        for load, n in enumerate(numbers):
            x = read(n, folder)
            hz = shaft_hz(n, load, folder)
            for i in range(len(x) // size):
                signals.append(x[i * size:(i + 1) * size])
                meta.append(dict(label=label, damage=damage, load=load, file=n, window=i, shaft_hz=hz))
    return np.array(signals), pd.DataFrame(meta)


def bandpass(x, f1, f2, fs=FS, order=4):
    sos = signal.butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def spectrum(x, fs=FS):
    """Amplitude spectrum with a Hann window."""
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft((x - x.mean()) * w)) * 2 / w.sum()
    return np.fft.rfftfreq(len(x), 1 / fs), X


def envelope_spectrum(x, f1=2000, f2=5500, fs=FS):
    """Band-pass the resonance band, demodulate with Hilbert, take the spectrum of the envelope."""
    return spectrum(np.abs(signal.hilbert(bandpass(x, f1, f2, fs))), fs)


def spectral_kurtosis(x, stft_windows=(64, 128, 256, 512, 1024), fs=FS):
    """STFT spectral kurtosis (Antoni): SK(f) = <|X|^4> / <|X|^2>^2 - 2.

    Impulsive bands score high and stationary Gaussian noise stays near zero. Each STFT window
    gives one row, which approximates reading a kurtogram.
    """
    rows = []
    for n in stft_windows:
        f, _, Z = signal.stft(x, fs=fs, nperseg=n, noverlap=3 * n // 4, boundary=None, padded=False)
        P = np.abs(Z) ** 2
        rows.append(pd.DataFrame(dict(window=n, f=f, sk=(P**2).mean(axis=1) / P.mean(axis=1) ** 2 - 2)))
    return pd.concat(rows, ignore_index=True)


def peak_ratio(x, order, shaft_hz, tol=0.02):
    """Envelope-spectrum peak near a fault frequency, relative to the spectrum's median floor."""
    f, E = envelope_spectrum(x)
    target = order * shaft_hz
    near = (f > target * (1 - tol)) & (f < target * (1 + tol))
    return E[near].max() / np.median(E[(f > 10) & (f < 500)])


def time_features(x):
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    return dict(rms=rms, peak=peak, crest=peak / rms, kurtosis=stats.kurtosis(x), skew=stats.skew(x),
                std=x.std(), shape=rms / np.mean(np.abs(x)), impulse=peak / np.mean(np.abs(x)))


def physics_features(x, shaft_hz, harmonics=3, tol=0.02):
    """Log peak-to-floor ratio at BPFO, BPFI and BSF harmonics; ignores absolute signal level."""
    f, E = envelope_spectrum(x)
    floor = np.median(E[(f > 10) & (f < 500)])
    out = {}
    for name, order in fault_orders().items():
        if name == "FTF":
            continue
        for h in range(1, harmonics + 1):
            target = h * order * shaft_hz
            near = (f > target * (1 - tol)) & (f < target * (1 + tol))
            out[f"{name}_h{h}"] = np.log(E[near].max() / floor)
    return out
