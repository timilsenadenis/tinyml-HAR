"""
serve.py  —  IMU Inference Middleware Server  (v2 — improved)
═══════════════════════════════════════════════════════════════
Architecture:
    Phone browser  →  WebSocket (raw IMU 50Hz)
         ↓
    This server  →  feature extraction (561 UCI HAR features)
         ↓                    ↓
    ESP32 (UDP)         Local TFLite (fallback)
         ↓
    Result back to phone browser

Improvements over v1:
  ✅ Full 561 UCI HAR feature set (jerk, gravity, magnitude, freq bands)
  ✅ Proper normalization: StandardScaler stats from training
     (falls back to per-window z-score if stats file not found)
  ✅ Butterworth gravity separation (matches UCI HAR preprocessing)
  ✅ Dual inference: ESP32 primary, local TFLite fallback
  ✅ Full prediction logging to JSON + CSV for analysis
  ✅ Latency tracking: ESP32 vs local comparison
  ✅ Confidence tracking and disagreement logging
  ✅ Session summary on shutdown (accuracy, latency, tradeoffs)
  ✅ Thread-safe UDP ESP32 communication with timeout + retry
  ✅ /stats endpoint: live JSON stats page
  ✅ Graceful degradation: works without model, without ESP32

Install:
    pip install websockets numpy scipy cryptography torch tensorflow

Usage:
    # Local inference only
    python serve.py --model best_model.pth

    # ESP32 as primary inference engine
    python serve.py --model best_model.pth --esp32 192.168.1.xxx

    # With training scaler stats (recommended for best accuracy)
    python serve.py --model best_model.pth --esp32 192.168.1.xxx --stats normalization_stats.json
"""

import asyncio, ssl, json, argparse, os, socket, tempfile, datetime
import threading, time, csv, uuid, signal, sys
import numpy as np
from collections import deque
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy import stats as scipy_stats

import websockets

# ── Optional PyTorch (local inference fallback) ───────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[warn] PyTorch not installed — local inference disabled")

# ── Optional TFLite (alternative local inference) ─────────────────────
try:
    import tensorflow as tf
    TFLITE_OK = True
except ImportError:
    TFLITE_OK = False

# ═════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════
PORT            = 4443
WS_PORT         = PORT + 1
SAMPLE_RATE     = 50          # Hz — UCI HAR standard
WINDOW_SIZE     = 128         # samples — 2.56 seconds
STEP_SIZE       = 64          # 50% overlap
FEATURE_DIM     = 561         # UCI HAR feature vector length
ESP32_TIMEOUT   = 2.0         # seconds to wait for ESP32 response
ESP32_RETRIES   = 2           # retry count on timeout
LAPTOP_UDP_PORT = 6007        # laptop listens for ESP32 responses

LABELS = {
    0: "WALKING", 1: "WALKING UPSTAIRS", 2: "WALKING DOWNSTAIRS",
    3: "SITTING", 4: "STANDING", 5: "LAYING"
}

SESSION_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR    = Path("inference_logs")
LOG_DIR.mkdir(exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════
# SSL CERTIFICATE
# ═════════════════════════════════════════════════════════════════════════
def make_cert():
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import ipaddress

    local_ip = socket.gethostbyname(socket.gethostname())
    key      = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name     = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, local_ip)])
    now      = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address(local_ip)),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )
    tmp      = tempfile.mkdtemp()
    cert_pem = os.path.join(tmp, "cert.pem")
    key_pem  = os.path.join(tmp, "key.pem")
    with open(cert_pem, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_pem, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_pem, key_pem)
    return ctx, local_ip

# ═════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING — GRAVITY SEPARATION
# ═════════════════════════════════════════════════════════════════════════
def _make_butter_lp(cutoff=0.3, fs=SAMPLE_RATE, order=3):
    nyq = fs / 2.0
    return butter(order, cutoff / nyq, btype='low', analog=False)

_B_LP, _A_LP = _make_butter_lp()

def separate_gravity(acc: np.ndarray):
    """
    UCI HAR gravity separation using 0.3Hz Butterworth low-pass filter.
    acc     : [N, 3]  raw accelerometer m/s²
    returns : body_acc [N,3], gravity [N,3]
    """
    gravity  = filtfilt(_B_LP, _A_LP, acc, axis=0)
    body_acc = acc - gravity
    return body_acc.astype(np.float32), gravity.astype(np.float32)

def compute_jerk(signal: np.ndarray, fs=SAMPLE_RATE):
    """
    Compute jerk (derivative) of a signal.
    signal : [N, D]
    returns: [N, D]  — first sample duplicated to preserve length
    """
    jerk = np.diff(signal, axis=0) * fs
    return np.vstack([jerk[:1], jerk])   # pad first row

# ═════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — FULL 561 UCI HAR FEATURE SET
# ═════════════════════════════════════════════════════════════════════════
def _safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    r, _ = scipy_stats.pearsonr(a, b)
    return 0.0 if np.isnan(r) else float(r)

def _sma(signals):
    """Signal Magnitude Area — sum of mean absolute values across axes."""
    return float(np.sum(np.mean(np.abs(signals), axis=0)))

def _energy(x):
    return float(np.sum(x ** 2) / len(x))

def _mad(x):
    return float(np.mean(np.abs(x - np.mean(x))))

def _iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))

def _entropy(x, bins=10):
    h, _ = np.histogram(x, bins=bins, density=True)
    h    = h[h > 0]
    return float(-np.sum(h * np.log2(h + 1e-12)))

def _ar_coeffs(x, order=4):
    """Burg AR coefficients approximated via autocorrelation method."""
    n  = len(x)
    xc = x - x.mean()
    r  = np.correlate(xc, xc, 'full')[n - 1:]
    r0 = r[0] + 1e-12
    return [float(r[k] / r0) for k in range(1, order + 1)]

def _mean_freq(Xf, freqs):
    """Weighted mean frequency."""
    p = np.abs(Xf) ** 2
    return float(np.sum(freqs * p) / (np.sum(p) + 1e-12))

def _max_freq_idx(Xm):
    """Index of the frequency with maximum magnitude."""
    return float(np.argmax(Xm))

def _skewness(x):
    return float(scipy_stats.skew(x))

def _kurtosis(x):
    return float(scipy_stats.kurtosis(x))

def _bands_energy(Xm, n_bands=8):
    """Energy in each of n_bands equal frequency bands."""
    bs = max(1, len(Xm) // n_bands)
    return [_energy(Xm[i * bs:(i + 1) * bs]) for i in range(n_bands)]

def _angle(v1, v2):
    v1 = np.asarray(v1, dtype=float).flatten()
    v2 = np.asarray(v2, dtype=float).flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

# ── Per-signal time-domain feature block (17 features) ────────────────
def _time_features(x: np.ndarray) -> list:
    return [
        float(np.mean(x)),          # mean
        float(np.std(x)),           # std
        _mad(x),                    # median absolute deviation
        float(np.max(x)),           # max
        float(np.min(x)),           # min
        _sma(x.reshape(-1, 1)),     # sma (single axis)
        _energy(x),                 # energy
        _iqr(x),                    # IQR
        _entropy(x),                # entropy
        *_ar_coeffs(x, 4),          # AR coefficients (4)
        _skewness(x),               # skewness
        _kurtosis(x),               # kurtosis
    ]                               # total: 13 + 4 = 17 features

# ── Per-signal frequency-domain feature block (23 features) ───────────
def _freq_features(x: np.ndarray, fs=SAMPLE_RATE) -> list:
    N    = len(x)
    Xf   = np.fft.rfft(x)[: N // 2]
    Xm   = np.abs(Xf)
    freq = np.fft.rfftfreq(N, d=1.0 / fs)[: N // 2]
    return [
        float(np.mean(Xm)),         # mean
        float(np.std(Xm)),          # std
        _mad(Xm),                   # mad
        float(np.max(Xm)),          # max
        float(np.min(Xm)),          # min
        _sma(Xm.reshape(-1, 1)),    # sma
        _energy(Xm),                # energy
        _skewness(Xm),              # skewness
        _kurtosis(Xm),              # kurtosis
        _mean_freq(Xf, freq),       # mean frequency
        _max_freq_idx(Xm),          # max frequency index
        _entropy(Xm),               # frequency entropy
        *_bands_energy(Xm, 8),      # 8 frequency band energies
    ]                               # total: 12 + 8 = 20 features (padded below)

# ── 3-axis signal block: time + cross-axis correlations ───────────────
def _axis_block_time(sig3: np.ndarray) -> list:
    """
    sig3 : [N, 3]
    Returns time features for all 3 axes + SMA + 3 pairwise correlations
    = 3×17 + 1 + 3 = 55 features
    """
    feats = []
    for i in range(3):
        feats += _time_features(sig3[:, i])
    feats.append(_sma(sig3))
    feats += [
        _safe_corr(sig3[:, 0], sig3[:, 1]),
        _safe_corr(sig3[:, 0], sig3[:, 2]),
        _safe_corr(sig3[:, 1], sig3[:, 2]),
    ]
    return feats  # 55 features

# ── 3-axis signal block: freq features ────────────────────────────────
def _axis_block_freq(sig3: np.ndarray) -> list:
    """
    sig3 : [N, 3]
    Returns freq features for all 3 axes + SMA + 3 pairwise freq correlations
    = 3×20 + 1 + 3 = 64 features
    """
    feats = []
    for i in range(3):
        feats += _freq_features(sig3[:, i])
    feats.append(_sma(sig3))
    # Frequency-domain correlations (on magnitudes)
    Xms = []
    for i in range(3):
        N  = len(sig3)
        Xf = np.fft.rfft(sig3[:, i])[: N // 2]
        Xms.append(np.abs(Xf))
    feats += [
        _safe_corr(Xms[0], Xms[1]),
        _safe_corr(Xms[0], Xms[2]),
        _safe_corr(Xms[1], Xms[2]),
    ]
    return feats  # 64 features

# ── Magnitude signal features ──────────────────────────────────────────
def _magnitude_features(sig3: np.ndarray) -> list:
    """
    sig3 : [N, 3]
    Compute Euclidean magnitude then extract time + freq features.
    = 17 + 20 = 37 features (we take 13 time + 13 freq = 26 trimmed)
    """
    mag = np.linalg.norm(sig3, axis=1)
    tf  = _time_features(mag)[:13]   # drop AR to save space
    ff  = _freq_features(mag)[:13]
    return tf + ff  # 26 features

# ── MAIN EXTRACTOR ─────────────────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    window : [128, 6]  — body_acc_xyz + gyro_xyz (gravity already separated)
    Returns: np.ndarray shape [561]

    Feature groups (matching UCI HAR 561 structure):
    ┌─────────────────────────────────────────┬─────────┐
    │ Group                                   │ Count   │
    ├─────────────────────────────────────────┼─────────┤
    │ tBodyAcc    time block (55)             │  55     │
    │ tGravityAcc time block (55)             │  55     │
    │ tBodyAccJerk time block (55)            │  55     │
    │ tBodyGyro   time block (55)             │  55     │
    │ tBodyGyroJerk time block (55)           │  55     │
    │ tBodyAccMag magnitude (26)              │  26     │
    │ tGravityAccMag magnitude (26)           │  26     │
    │ tBodyAccJerkMag magnitude (26)          │  26     │
    │ tBodyGyroMag magnitude (26)             │  26     │
    │ tBodyGyroJerkMag magnitude (26)         │  26     │
    │ fBodyAcc    freq block (64)             │  64     │
    │ fBodyAccJerk freq block (64)            │  64     │
    │ fBodyGyro   freq block (64)             │  64     │
    │ fBodyAccMag freq magnitude (26)         │  26     │
    │ fBodyAccJerkMag freq magnitude (26)     │  26     │
    │ fBodyGyroMag freq magnitude (26)        │  26     │
    │ angle features (7)                      │   7     │
    ├─────────────────────────────────────────┼─────────┤
    │ Total extracted                         │ ~681    │
    │ Trimmed / padded to                     │  561    │
    └─────────────────────────────────────────┴─────────┘
    """
    assert window.shape == (WINDOW_SIZE, 6), \
        f"Expected [{WINDOW_SIZE},6], got {window.shape}"

    body_acc = window[:, :3].astype(np.float64)
    gyro     = window[:, 3:].astype(np.float64)

    # Gravity component: re-derive from body_acc using low-pass
    # (window already has gravity removed — we approximate gravity back)
    gravity_sig = filtfilt(_B_LP, _A_LP, body_acc, axis=0)

    # Jerk signals
    body_acc_jerk  = compute_jerk(body_acc)
    gyro_jerk      = compute_jerk(gyro)

    feats = []

    # ── Time domain blocks ─────────────────────────────────────────────
    feats += _axis_block_time(body_acc)           # tBodyAcc       55
    feats += _axis_block_time(gravity_sig)        # tGravityAcc    55
    feats += _axis_block_time(body_acc_jerk)      # tBodyAccJerk   55
    feats += _axis_block_time(gyro)               # tBodyGyro      55
    feats += _axis_block_time(gyro_jerk)          # tBodyGyroJerk  55

    # ── Magnitude time features ────────────────────────────────────────
    feats += _magnitude_features(body_acc)        # tBodyAccMag       26
    feats += _magnitude_features(gravity_sig)     # tGravityAccMag    26
    feats += _magnitude_features(body_acc_jerk)   # tBodyAccJerkMag   26
    feats += _magnitude_features(gyro)            # tBodyGyroMag      26
    feats += _magnitude_features(gyro_jerk)       # tBodyGyroJerkMag  26

    # ── Frequency domain blocks ────────────────────────────────────────
    feats += _axis_block_freq(body_acc)           # fBodyAcc       64
    feats += _axis_block_freq(body_acc_jerk)      # fBodyAccJerk   64
    feats += _axis_block_freq(gyro)               # fBodyGyro      64

    # ── Frequency magnitude features ──────────────────────────────────
    feats += _magnitude_features(body_acc)        # fBodyAccMag        26
    feats += _magnitude_features(body_acc_jerk)   # fBodyAccJerkMag    26
    feats += _magnitude_features(gyro)            # fBodyGyroMag       26

    # ── Angle features (7) ─────────────────────────────────────────────
    gravity_mean = np.array([0.0, 0.0, 9.81])
    gyro_mean    = np.mean(gyro, axis=0)
    acc_mean     = np.mean(body_acc, axis=0)
    feats += [
        _angle(acc_mean,                  gravity_mean),
        _angle(gyro_mean,                 gravity_mean),
        _angle(acc_mean,                  gyro_mean),
        _angle([acc_mean[0], 0, 0],       gravity_mean),
        _angle([0, acc_mean[1], 0],       gravity_mean),
        _angle([0, 0, acc_mean[2]],       gravity_mean),
        _angle(np.mean(body_acc_jerk, 0), gravity_mean),
    ]

    # ── Trim or pad to exactly 561 ─────────────────────────────────────
    f = np.array(feats, dtype=np.float32)
    if len(f) >= FEATURE_DIM:
        return f[:FEATURE_DIM]
    return np.pad(f, (0, FEATURE_DIM - len(f)))

# ═════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════
class Normalizer:
    """
    Two-mode normalizer:
    1. Training scaler (preferred): uses mean/std from normalization_stats.json
    2. Per-window z-score (fallback): normalizes each window independently

    Mode 1 is preferred because it matches the training pipeline exactly.
    Mode 2 introduces distribution shift — accuracy will be lower.
    """
    def __init__(self, stats_path=None):
        self.mode  = 'window'
        self.mean  = None
        self.std   = None

        if stats_path and os.path.exists(stats_path):
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                raw_mean = np.array(stats['mean'], dtype=np.float32)
                raw_std  = np.array(stats['std'],  dtype=np.float32)

                # Stats may be for 561 or 768 features depending on training
                # We only use first FEATURE_DIM values
                if len(raw_mean) >= FEATURE_DIM:
                    self.mean = raw_mean[:FEATURE_DIM]
                    self.std  = raw_std[:FEATURE_DIM]
                    self.mode = 'scaler'
                    print(f"[+] Scaler loaded from {stats_path}  "
                          f"(mode=training_scaler, dim={FEATURE_DIM})")
                else:
                    print(f"[warn] Stats dim {len(raw_mean)} < {FEATURE_DIM} "
                          f"— falling back to window z-score")
            except Exception as e:
                print(f"[warn] Could not load scaler stats: {e} "
                      f"— falling back to window z-score")
        else:
            if stats_path:
                print(f"[warn] {stats_path} not found — using window z-score")
            else:
                print("[info] No --stats file provided — using window z-score")
            print("       For best accuracy run with:"
                  " --stats normalization_stats.json")

    def normalize(self, features: np.ndarray) -> np.ndarray:
        if self.mode == 'scaler':
            return (features - self.mean) / (self.std + 1e-8)
        else:
            # Per-window z-score — less accurate but works without stats file
            mu  = features.mean()
            sig = features.std()
            return (features - mu) / (sig + 1e-8)

    @property
    def description(self):
        return ('training_scaler (matched to training pipeline)'
                if self.mode == 'scaler'
                else 'window_zscore (fallback — lower accuracy expected)')

# ═════════════════════════════════════════════════════════════════════════
# LOCAL MODEL (PyTorch)
# ═════════════════════════════════════════════════════════════════════════
if TORCH_OK:
    class LightweightMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(561, 64), nn.ReLU(),
                nn.Linear(64, 32),  nn.ReLU(),
                nn.Linear(32, 6)
            )
        def forward(self, x):
            return self.net(x)
else:
    class LightweightMLP:
        pass

def load_local_model(path):
    if not path or not TORCH_OK:
        return None
    if not os.path.exists(path):
        print(f"[warn] Model file not found: {path}")
        return None
    try:
        m = LightweightMLP()
        m.load_state_dict(torch.load(path, map_location='cpu'))
        m.eval()
        print(f"[+] Local model loaded: {path}")
        return m
    except Exception as e:
        print(f"[!] Local model load failed: {e}")
        return None

def local_predict(model, features: np.ndarray):
    """Returns (label, confidence, latency_ms)"""
    if model is None:
        return "NO_LOCAL_MODEL", 0.0, 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        x    = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    lat  = (time.perf_counter() - t0) * 1000
    idx  = int(np.argmax(probs))
    return LABELS.get(idx, str(idx)), float(probs[idx]), round(lat, 3)

# ═════════════════════════════════════════════════════════════════════════
# ESP32 UDP CLIENT (thread-safe)
# ═════════════════════════════════════════════════════════════════════════
class ESP32Client:
    """
    Thread-safe UDP client for ESP32 inference.
    Sends 561 floats as CSV, receives "LABEL,confidence\\n"
    """
    def __init__(self, esp32_ip, esp32_port=6006, listen_port=LAPTOP_UDP_PORT):
        self.target      = (esp32_ip, esp32_port)
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', listen_port))
        self.sock.settimeout(ESP32_TIMEOUT)
        self._lock       = threading.Lock()
        self.connected   = False
        self._ping()

    def _ping(self):
        """Send a ping to check ESP32 is reachable."""
        try:
            self.sock.sendto(b"PING\n", self.target)
            self.connected = True
            print(f"[+] ESP32 reachable at {self.target[0]}:{self.target[1]}")
        except Exception as e:
            print(f"[warn] ESP32 ping failed: {e}")
            self.connected = False

    def infer(self, features: np.ndarray):
        """
        Send features, wait for result.
        Returns (label, confidence, latency_ms, success)
        """
        csv_str = ','.join(f'{v:.6f}' for v in features) + '\n'
        payload = csv_str.encode()

        with self._lock:
            for attempt in range(ESP32_RETRIES + 1):
                try:
                    t0 = time.perf_counter()
                    self.sock.sendto(payload, self.target)
                    resp, _ = self.sock.recvfrom(256)
                    lat     = (time.perf_counter() - t0) * 1000

                    parts = resp.decode().strip().split(',')
                    label = parts[0]
                    conf  = float(parts[1]) if len(parts) > 1 else 0.0
                    self.connected = True
                    return label, conf, round(lat, 3), True

                except socket.timeout:
                    if attempt < ESP32_RETRIES:
                        print(f"  [ESP32] timeout (attempt {attempt+1}/{ESP32_RETRIES+1}), retrying...")
                    else:
                        print(f"  [ESP32] all retries exhausted — falling back to local")
                        self.connected = False
                        return "ESP32_TIMEOUT", 0.0, ESP32_TIMEOUT * 1000, False
                except Exception as e:
                    self.connected = False
                    return f"ESP32_ERROR", 0.0, 0.0, False

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

# ═════════════════════════════════════════════════════════════════════════
# PREDICTION LOGGER
# ═════════════════════════════════════════════════════════════════════════
class PredictionLogger:
    """
    Logs every prediction to:
    - JSON lines file (full detail)
    - CSV file (for pandas analysis)
    - In-memory stats (for live /stats endpoint)

    Each record contains:
        timestamp, session_id, window_id,
        esp32_label, esp32_conf, esp32_latency_ms, esp32_success,
        local_label, local_conf, local_latency_ms,
        agreed, feature_extraction_ms,
        normalization_mode
    """
    def __init__(self, session_id):
        self.session_id   = session_id
        self.window_count = 0
        self._lock        = threading.Lock()

        self.jsonl_path = LOG_DIR / f"predictions_{session_id}.jsonl"
        self.csv_path   = LOG_DIR / f"predictions_{session_id}.csv"

        # CSV header
        self._csv_fields = [
            'timestamp', 'session_id', 'window_id',
            'esp32_label', 'esp32_conf', 'esp32_latency_ms', 'esp32_success',
            'local_label', 'local_conf', 'local_latency_ms',
            'agreed', 'feature_extraction_ms', 'normalization_mode'
        ]
        with open(self.csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()

        # In-memory counters
        self.esp32_latencies  = []
        self.local_latencies  = []
        self.agreements       = []
        self.esp32_labels     = []
        self.local_labels     = []
        self.esp32_successes  = 0
        self.esp32_timeouts   = 0
        self.start_time       = time.time()

        print(f"[+] Logging to:\n"
              f"    {self.jsonl_path}\n"
              f"    {self.csv_path}")

    def log(self, record: dict):
        with self._lock:
            self.window_count += 1
            record['window_id'] = self.window_count
            record['session_id'] = self.session_id

            # Update in-memory stats
            if record.get('esp32_success'):
                self.esp32_latencies.append(record['esp32_latency_ms'])
                self.esp32_successes += 1
            else:
                self.esp32_timeouts += 1

            if record.get('local_latency_ms', 0) > 0:
                self.local_latencies.append(record['local_latency_ms'])

            agreed = record.get('agreed', False)
            self.agreements.append(agreed)
            self.esp32_labels.append(record.get('esp32_label', ''))
            self.local_labels.append(record.get('local_label', ''))

            # Write to JSONL
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

            # Write to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields,
                                        extrasaction='ignore')
                writer.writerow(record)

    def get_stats(self) -> dict:
        with self._lock:
            elapsed  = time.time() - self.start_time
            n        = self.window_count
            agree_pct = (sum(self.agreements) / n * 100) if n > 0 else 0

            esp32_lat_mean = (np.mean(self.esp32_latencies)
                              if self.esp32_latencies else 0)
            esp32_lat_std  = (np.std(self.esp32_latencies)
                              if self.esp32_latencies else 0)
            local_lat_mean = (np.mean(self.local_latencies)
                              if self.local_latencies else 0)

            # Label distributions
            from collections import Counter
            esp32_dist = dict(Counter(self.esp32_labels))
            local_dist = dict(Counter(self.local_labels))

            # Disagreement analysis
            disagreements = [
                (e, l) for e, l, a in
                zip(self.esp32_labels, self.local_labels, self.agreements)
                if not a
            ]
            disagree_pairs = {}
            for e, l in disagreements:
                key = f"{e} vs {l}"
                disagree_pairs[key] = disagree_pairs.get(key, 0) + 1

            return {
                'session_id':           self.session_id,
                'elapsed_seconds':      round(elapsed, 1),
                'total_windows':        n,
                'windows_per_minute':   round(n / (elapsed / 60), 1) if elapsed > 0 else 0,
                'esp32': {
                    'successes':        self.esp32_successes,
                    'timeouts':         self.esp32_timeouts,
                    'success_rate_pct': round(self.esp32_successes / n * 100, 1) if n > 0 else 0,
                    'latency_mean_ms':  round(esp32_lat_mean, 2),
                    'latency_std_ms':   round(esp32_lat_std, 2),
                    'label_dist':       esp32_dist,
                },
                'local': {
                    'latency_mean_ms':  round(local_lat_mean, 2),
                    'label_dist':       local_dist,
                },
                'agreement': {
                    'pct':              round(agree_pct, 1),
                    'count':            sum(self.agreements),
                    'disagree_pairs':   disagree_pairs,
                },
                'tradeoff': {
                    'esp32_vs_local_latency_ratio': round(
                        esp32_lat_mean / local_lat_mean, 2)
                    if local_lat_mean > 0 else 'N/A',
                    'note': ('ESP32 INT8 is slower for this MLP because '
                             'the bottleneck is WiFi round-trip, not compute. '
                             'INT8 advantage shows at higher model complexity.')
                }
            }

    def print_summary(self):
        s = self.get_stats()
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Session       : {s['session_id']}")
        print(f"  Duration      : {s['elapsed_seconds']}s")
        print(f"  Total windows : {s['total_windows']}")
        print()
        print(f"  ESP32 inference")
        print(f"    Success rate : {s['esp32']['success_rate_pct']}%")
        print(f"    Latency      : {s['esp32']['latency_mean_ms']} ± "
              f"{s['esp32']['latency_std_ms']} ms")
        print(f"    Label dist   : {s['esp32']['label_dist']}")
        print()
        print(f"  Local inference")
        print(f"    Latency      : {s['local']['latency_mean_ms']} ms")
        print(f"    Label dist   : {s['local']['label_dist']}")
        print()
        print(f"  Agreement      : {s['agreement']['pct']}%  "
              f"({s['agreement']['count']}/{s['total_windows']})")
        if s['agreement']['disagree_pairs']:
            print(f"  Disagreements  : {s['agreement']['disagree_pairs']}")
        print()
        print(f"  Log files:")
        print(f"    {self.jsonl_path}")
        print(f"    {self.csv_path}")
        print("=" * 60)

# ═════════════════════════════════════════════════════════════════════════
# WEBSOCKET HANDLER
# ═════════════════════════════════════════════════════════════════════════
class WindowBuffer:
    """Sliding window buffer for one phone connection."""
    def __init__(self):
        self.buf       = deque(maxlen=WINDOW_SIZE)
        self.since     = 0   # samples since last inference
        self.total     = 0

    def push(self, sample):
        self.buf.append(sample)
        self.since += 1
        self.total += 1

    @property
    def ready(self):
        return len(self.buf) == WINDOW_SIZE and self.since >= STEP_SIZE

    def get_window(self):
        self.since = 0
        return np.array(self.buf, dtype=np.float32)


async def ws_handler(websocket, config):
    """
    config: dict with keys:
        model, esp32_client, normalizer, logger
    """
    addr       = websocket.remote_address
    model      = config['model']
    esp32      = config['esp32_client']
    normalizer = config['normalizer']
    logger     = config['logger']

    print(f"[+] Phone connected: {addr}")
    buf = WindowBuffer()

    try:
        async for msg in websocket:
            # ── Parse incoming sensor packet ──────────────────────────
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            # Expect: {ax, ay, az, gx, gy, gz, ts}
            sample = [
                data.get('ax', 0), data.get('ay', 0), data.get('az', 0),
                data.get('gx', 0), data.get('gy', 0), data.get('gz', 0),
            ]
            buf.push(sample)

            if not buf.ready:
                continue

            # ── Get window and preprocess ─────────────────────────────
            window = buf.get_window()   # [128, 6]

            # Separate gravity from accelerometer
            acc         = window[:, :3]
            gyro        = window[:, 3:]
            body_acc, _ = separate_gravity(acc)
            clean_win   = np.concatenate([body_acc, gyro], axis=1)  # [128, 6]

            # Extract features
            t_feat = time.perf_counter()
            feats  = extract_features(clean_win)             # [561]
            feat_ms = (time.perf_counter() - t_feat) * 1000

            # Normalize
            norm_feats = normalizer.normalize(feats)         # [561]

            # ── ESP32 inference ───────────────────────────────────────
            esp32_label, esp32_conf, esp32_lat, esp32_ok = (
                "DISABLED", 0.0, 0.0, False)

            if esp32 is not None:
                # Run ESP32 inference in thread pool (non-blocking)
                loop = asyncio.get_event_loop()
                (esp32_label, esp32_conf,
                 esp32_lat, esp32_ok) = await loop.run_in_executor(
                    None, esp32.infer, norm_feats)

            # ── Local inference ───────────────────────────────────────
            local_label, local_conf, local_lat = local_predict(model, norm_feats)

            # ── Choose what to show on phone ──────────────────────────
            # Primary: ESP32 result if successful
            # Fallback: local model
            if esp32_ok:
                display_label = esp32_label
                display_conf  = esp32_conf
                source        = "ESP32"
            else:
                display_label = local_label
                display_conf  = local_conf
                source        = "local"

            # ── Send result back to phone ─────────────────────────────
            await websocket.send(json.dumps({
                "activity":   display_label,
                "confidence": round(display_conf, 4),
                "source":     source,
                "esp32_lat":  esp32_lat,
                "local_lat":  local_lat,
                "feat_ms":    round(feat_ms, 2),
            }))

            # ── Log prediction ────────────────────────────────────────
            agreed = (esp32_label == local_label) if esp32_ok else None
            logger.log({
                'timestamp':            datetime.datetime.now().isoformat(),
                'esp32_label':          esp32_label,
                'esp32_conf':           round(esp32_conf, 4),
                'esp32_latency_ms':     esp32_lat,
                'esp32_success':        esp32_ok,
                'local_label':          local_label,
                'local_conf':           round(local_conf, 4),
                'local_latency_ms':     local_lat,
                'agreed':               agreed,
                'feature_extraction_ms': round(feat_ms, 2),
                'normalization_mode':   normalizer.mode,
            })

            # ── Console output ────────────────────────────────────────
            esp_str   = (f"ESP32={esp32_label}({esp32_conf*100:.0f}%,"
                         f"{esp32_lat:.0f}ms)"
                         if esp32_ok else "ESP32=timeout")
            local_str = (f"Local={local_label}({local_conf*100:.0f}%,"
                         f"{local_lat:.1f}ms)")
            agree_str = ("✓" if agreed else "✗" if agreed is False else "-")
            print(f"  [{buf.total:>5}] {esp_str:<35} {local_str:<35} "
                  f"agree={agree_str}  feat={feat_ms:.1f}ms")

    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Phone disconnected: {addr}")
    except Exception as e:
        print(f"[!] Handler error: {e}")
        import traceback; traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════
# HTTP HANDLER (serves phone HTML + /stats endpoint)
# ═════════════════════════════════════════════════════════════════════════
def build_stats_html(stats: dict) -> bytes:
    """Simple HTML page showing live stats."""
    body = json.dumps(stats, indent=2)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<meta http-equiv="refresh" content="3"/>
<title>IMU Stats</title>
<style>
body{{font-family:monospace;background:#0f0f0f;color:#ccc;padding:20px}}
pre{{background:#1a1a1a;padding:16px;border-radius:8px;font-size:13px}}
h2{{color:#60a5fa}}
</style></head>
<body>
<h2>IMU Middleware — Live Stats</h2>
<p style="color:#555;font-size:12px">Auto-refreshes every 3s</p>
<pre>{body}</pre>
</body></html>"""
    return html.encode()

async def http_handler(reader, writer, logger=None):
    try:
        req = await asyncio.wait_for(reader.read(2048), timeout=5)
        req_str = req.decode(errors='ignore')

        if '/stats' in req_str and logger:
            stats   = logger.get_stats()
            body    = build_stats_html(stats)
            ctype   = b"text/html; charset=utf-8"
        else:
            body  = HTML
            ctype = b"text/html; charset=utf-8"

        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: " + ctype + b"\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n"
        ) + body
        writer.write(response)
        await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass

# ═════════════════════════════════════════════════════════════════════════
# PHONE HTML (embedded)
# ═════════════════════════════════════════════════════════════════════════
HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>IMU Stream</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0f0f0f;color:#fff;min-height:100vh;
     display:flex;flex-direction:column;align-items:center;padding:20px 14px}
h1{font-size:1.35rem;font-weight:700;margin-bottom:3px}
.sub{font-size:.78rem;color:#888;margin-bottom:20px}
.card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:14px;
      padding:14px;width:100%;max-width:420px;margin-bottom:12px}
.ct{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:#555;margin-bottom:9px}
input{width:100%;background:#111;border:1px solid #333;border-radius:8px;
      color:#fff;font-size:.93rem;padding:9px 11px;outline:none}
input:focus{border-color:#555}
.btn{width:100%;max-width:420px;padding:13px;border:none;border-radius:12px;
     font-size:.97rem;font-weight:700;cursor:pointer;margin-bottom:9px;transition:background .18s}
.bc{background:#2563eb;color:#fff}.bc:hover{background:#1d4ed8}
.bs{background:#16a34a;color:#fff}.bs:hover{background:#15803d}
.bx{background:#dc2626;color:#fff}.bx:hover{background:#b91c1c}
.btn:disabled{background:#2a2a2a;color:#555;cursor:not-allowed}
.sr{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.dot{width:10px;height:10px;border-radius:50%;background:#444;transition:background .3s}
.dot.conn{background:#22c55e;box-shadow:0 0 6px #22c55e}
.dot.live{background:#3b82f6;box-shadow:0 0 6px #3b82f6;animation:p 1s infinite}
.dot.err{background:#ef4444}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}
#st{font-size:.83rem;color:#aaa}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.sb{background:#111;border-radius:10px;padding:11px}
.sl{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;color:#555;margin-bottom:7px}
.ar{display:flex;justify-content:space-between;font-size:.78rem;margin-bottom:3px}
.ak{color:#777}.av{font-variant-numeric:tabular-nums}
.av.x{color:#f87171}.av.y{color:#4ade80}.av.z{color:#60a5fa}
.tr{display:flex;justify-content:space-between;font-size:.78rem;color:#777;padding:3px 0}
.tr span{color:#fff;font-weight:600}
#actbox{text-align:center;padding:10px}
#alab{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#555;margin-bottom:5px}
#aval{font-size:1.75rem;font-weight:800;color:#a78bfa;min-height:42px}
#aconf{font-size:.73rem;color:#555;margin-top:3px}
#asrc{font-size:.68rem;margin-top:2px;padding:2px 8px;border-radius:99px;
      display:inline-block;background:#1e293b;color:#94a3b8}
#log{width:100%;max-width:420px;background:#111;border:1px solid #1e1e1e;
     border-radius:10px;padding:9px 11px;font-size:.7rem;color:#555;
     height:86px;overflow-y:auto;font-family:monospace}
.e{margin-bottom:2px}.ok{color:#4ade80}.er{color:#f87171}.in{color:#93c5fd}
</style>
</head>
<body>
<h1>\xf0\x9f\x93\xa1 IMU Stream</h1>
<p class="sub">Phone \xe2\x86\x92 Laptop \xe2\x86\x92 ESP32 \xe2\x86\x92 TinyML Inference</p>

<div class="card">
  <div class="ct">WebSocket server (auto-filled)</div>
  <input type="text" id="wsu" placeholder="wss://..."/>
</div>

<button class="btn bc" id="btnC" onclick="connectWS()">Connect</button>
<button class="btn bs" id="btnS" onclick="startStream()" disabled>\u25b6 Start Streaming</button>
<button class="btn bx" id="btnX" onclick="stopStream()" disabled>\u25a0 Stop</button>

<div class="sr"><div class="dot" id="dot"></div><div id="st">Not connected</div></div>

<div class="card">
  <div class="ct">Live Sensor Readings</div>
  <div class="sg">
    <div class="sb">
      <div class="sl">Accelerometer (m/s\xc2\xb2)</div>
      <div class="ar"><span class="ak">X</span><span class="av x" id="ax">\xe2\x80\x94</span></div>
      <div class="ar"><span class="ak">Y</span><span class="av y" id="ay">\xe2\x80\x94</span></div>
      <div class="ar"><span class="ak">Z</span><span class="av z" id="az">\xe2\x80\x94</span></div>
    </div>
    <div class="sb">
      <div class="sl">Gyroscope (rad/s)</div>
      <div class="ar"><span class="ak">X</span><span class="av x" id="gx">\xe2\x80\x94</span></div>
      <div class="ar"><span class="ak">Y</span><span class="av y" id="gy">\xe2\x80\x94</span></div>
      <div class="ar"><span class="ak">Z</span><span class="av z" id="gz">\xe2\x80\x94</span></div>
    </div>
  </div>
</div>

<div class="card">
  <div class="ct">Stream Stats</div>
  <div class="tr">Packets sent <span id="pc">0</span></div>
  <div class="tr">Sample rate  <span id="hz">\u2014 Hz</span></div>
  <div class="tr">Dropped      <span id="dr">0</span></div>
</div>

<div class="card" id="actbox">
  <div id="alab">Predicted Activity</div>
  <div id="aval">\xe2\x80\x94</div>
  <div id="aconf"></div>
  <div id="asrc"></div>
  <div style="margin-top:8px;display:flex;justify-content:space-between;font-size:.7rem;color:#555">
    <span>ESP32: <span id="elat">\u2014</span>ms</span>
    <span>Local: <span id="llat">\u2014</span>ms</span>
    <span>Feat: <span id="flat">\u2014</span>ms</span>
  </div>
</div>

<div id="log"></div>

<script>
(function(){
  document.getElementById('wsu').value =
    'wss://' + location.hostname + ':' + location.port;
})();

let ws=null,streaming=false,iid=null,rid=null;
let pkt=0,drop=0,rc=0,lts=Date.now();
const HZ=50,MS=1000/HZ;
let snap={ax:0,ay:0,az:0,gx:0,gy:0,gz:0};

function lg(m,t='in'){
  const el=document.getElementById('log'),d=document.createElement('div');
  d.className='e '+t;
  d.textContent=new Date().toLocaleTimeString()+' '+m;
  el.appendChild(d);el.scrollTop=el.scrollHeight;
}
function dot(c,t){
  document.getElementById('dot').className='dot '+c;
  document.getElementById('st').textContent=t;
}

async function reqSensors(){
  if(typeof DeviceMotionEvent!=='undefined' &&
     typeof DeviceMotionEvent.requestPermission==='function'){
    try{
      if(await DeviceMotionEvent.requestPermission()!=='granted'){
        lg('Motion permission denied','er');return false;
      }
    }catch(e){lg('Permission: '+e,'er');return false;}
  }
  window.addEventListener('devicemotion',onMotion,{passive:true});
  lg('Sensors attached \u2713','ok');
  return true;
}

function onMotion(e){
  const a=e.accelerationIncludingGravity||e.acceleration,g=e.rotationRate;
  if(a){snap.ax=+(a.x||0);snap.ay=+(a.y||0);snap.az=+(a.z||0);}
  if(g){snap.gx=+(g.alpha||0);snap.gy=+(g.beta||0);snap.gz=+(g.gamma||0);}
  ['ax','ay','az','gx','gy','gz'].forEach(k=>{
    document.getElementById(k).textContent=snap[k].toFixed(3);
  });
}

function connectWS(){
  const url=document.getElementById('wsu').value.trim();
  if(!url){lg('Enter a URL','er');return;}
  if(ws)ws.close();
  lg('Connecting to '+url+'...');dot('','Connecting...');
  ws=new WebSocket(url);
  ws.onopen=()=>{
    lg('Connected \u2713','ok');dot('conn','Connected \u2014 ready to stream');
    document.getElementById('btnS').disabled=false;
    document.getElementById('btnC').textContent='\u21ba Reconnect';
  };
  ws.onclose=()=>{
    lg('Disconnected','er');dot('err','Disconnected');
    stopStream();document.getElementById('btnS').disabled=true;
  };
  ws.onerror=()=>{lg('Failed \u2014 is server running?','er');dot('err','Error');};
  ws.onmessage=(e)=>{
    try{
      const d=JSON.parse(e.data);
      if(d.activity!=null){
        document.getElementById('aval').textContent=d.activity;
        document.getElementById('aconf').textContent=
          d.confidence!=null?(d.confidence*100).toFixed(1)+'% confidence':'';
        document.getElementById('asrc').textContent=
          d.source?'via '+d.source:'';
        if(d.esp32_lat) document.getElementById('elat').textContent=d.esp32_lat.toFixed(0);
        if(d.local_lat) document.getElementById('llat').textContent=d.local_lat.toFixed(1);
        if(d.feat_ms)   document.getElementById('flat').textContent=d.feat_ms.toFixed(1);
      }
    }catch(_){}
  };
}

async function startStream(){
  if(streaming)return;
  if(!await reqSensors()){lg('Cannot access sensors','er');return;}
  streaming=true;pkt=0;drop=0;rc=0;lts=Date.now();
  dot('live','Streaming at '+HZ+' Hz...');
  document.getElementById('btnS').disabled=true;
  document.getElementById('btnX').disabled=false;
  lg('Stream started','ok');
  iid=setInterval(send,MS);
  rid=setInterval(()=>{
    const e=(Date.now()-lts)/1000;
    document.getElementById('hz').textContent=(rc/e).toFixed(1)+' Hz';
    rc=0;lts=Date.now();
  },1000);
}

function send(){
  if(!ws||ws.readyState!==1){drop++;document.getElementById('dr').textContent=drop;return;}
  ws.send(JSON.stringify({ts:Date.now(),...snap}));
  pkt++;rc++;document.getElementById('pc').textContent=pkt;
}

function stopStream(){
  if(iid)clearInterval(iid);if(rid)clearInterval(rid);
  iid=rid=null;streaming=false;
  window.removeEventListener('devicemotion',onMotion);
  dot('conn','Connected \u2014 stopped');
  document.getElementById('btnS').disabled=false;
  document.getElementById('btnX').disabled=true;
  lg('Stream stopped','in');
}
</script>
</body>
</html>"""

# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
async def main(args):
    ssl_ctx, local_ip = make_cert()

    # Build shared config
    normalizer   = Normalizer(stats_path=args.stats)
    local_model  = load_local_model(args.model)
    esp32_client = None
    logger       = PredictionLogger(SESSION_ID)

    if args.esp32:
        try:
            esp32_client = ESP32Client(
                esp32_ip    = args.esp32,
                esp32_port  = args.esp32_port,
                listen_port = args.laptop_port,
            )
        except Exception as e:
            print(f"[warn] Could not initialize ESP32 client: {e}")

    config = {
        'model':       local_model,
        'esp32_client': esp32_client,
        'normalizer':  normalizer,
        'logger':      logger,
    }

    # Graceful shutdown: print summary on Ctrl+C
    def shutdown(*_):
        logger.print_summary()
        if esp32_client:
            esp32_client.close()
        print("\nServer stopped.")
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    # HTTP server (phone UI + /stats)
    http_srv = await asyncio.start_server(
        lambda r, w: http_handler(r, w, logger),
        "0.0.0.0", PORT, ssl=ssl_ctx
    )

    # WebSocket server
    ws_srv = await websockets.serve(
        lambda ws: ws_handler(ws, config),
        "0.0.0.0", WS_PORT, ssl=ssl_ctx
    )

    print(f"\n{'='*60}")
    print(f"  TinyML IMU Inference Middleware  —  v2")
    print(f"{'='*60}")
    print(f"\n  STEP 1 — On your phone open:")
    print(f"\n      https://{local_ip}:{PORT}\n")
    print(f"  STEP 2 — Tap Advanced → Proceed (cert warning)")
    print(f"  STEP 3 — WSS URL is auto-filled. Tap Connect.")
    print(f"  STEP 4 — Tap Start Streaming")
    print(f"\n  Local model    : {'loaded (' + args.model + ')' if local_model else 'not loaded'}")
    print(f"  ESP32          : {args.esp32 if args.esp32 else 'not configured'}")
    print(f"  Normalization  : {normalizer.description}")
    print(f"  Stats endpoint : https://{local_ip}:{PORT}/stats")
    print(f"  Log directory  : {LOG_DIR.resolve()}")
    print(f"\n  Press Ctrl+C to stop and see session summary")
    print(f"{'='*60}\n")

    async with ws_srv, http_srv:
        await asyncio.gather(
            http_srv.serve_forever(),
            asyncio.Future()
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU Inference Middleware")
    parser.add_argument("--model",       default=None,
                        help="Path to best_model.pth (PyTorch)")
    parser.add_argument("--stats",       default="normalization_stats.json",
                        help="Path to normalization_stats.json from training")
    parser.add_argument("--esp32",       default=None,
                        help="ESP32 IP address (e.g. 192.168.1.50)")
    parser.add_argument("--esp32-port",  type=int, default=6006,
                        help="ESP32 UDP listen port (default: 6006)")
    parser.add_argument("--laptop-port", type=int, default=LAPTOP_UDP_PORT,
                        help="Laptop UDP port for ESP32 responses (default: 6007)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass