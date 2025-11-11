"""
prep_pamap2_to_npy.py

Make PAMAP2 -> .npy files that match the provided base.yaml WITHOUT editing YAML.

- Reads Hydra-style base.yaml to get:
  * dataset.modalities (e.g., [imu_hand, imu_chest, imu_ankle, heart_rate])
  * dataset train/val/test split ratios
- Produces per-modality arrays whose last dim == 64 (as model.encoders.imu.input_dim=64
  and train.py defaults to 64 when per-modality is not specified).
- NO zero-padding: (T,6)/(T,1) are mapped to (T,64) via meaningful feature engineering.

Pipeline
1) Load raw .dat (54 cols): timestamp, activityId, heart_rate, and 3x IMU blocks (hand/chest/ankle).
2) Select 6-axis IMU channels (ACC±16g 3 + GYRO 3). (±6g/orientation excluded)
3) Resample all to a common target Hz (default 100 Hz) with interpolation + NaN handling.
4) Windowing (default 5s window, 2.5s hop). Majority label per window; drop windows with label 0.
5) For each window:
   - IMU: (T,6) -> (T,64) via multiscale rolling stats, diff, magnitudes etc.
   - HR : (T,1) -> (T,64) via lags, rolling stats, EWMA, diffs, slopes, etc.
6) Subject-wise split to train/val/test using yaml ratios.
7) Save per-split:
   {modality}.npy  (N, T, 64)
   labels.npy      (N,)
   subjects.npy    (N,)
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


# =========================
# YAML / Config
# =========================
def load_cfg(path: Path):
    return OmegaConf.load(path)


def get_target_dim_for_mod(cfg, modality: str) -> int:
    """
    Match train.py behavior:
    - If encoders.<modality>.input_dim exists -> use it.
    - Else if 'imu_*' and encoders.imu.input_dim exists -> use it (64 in the provided YAML).
    - Else default to 64 (train.py uses 64 as fallback).
    """
    enc = getattr(cfg.model, "encoders", {})
    if modality in enc and "input_dim" in enc[modality]:
        return int(enc[modality]["input_dim"])
    if modality.startswith("imu_") and "imu" in enc and "input_dim" in enc["imu"]:
        return int(enc["imu"]["input_dim"])
    return 64


# =========================
# PAMAP2 schema
# =========================
# 1:timestamp, 2:activityId, 3:heart_rate,
# hand(4-20) chest(21-37) ankle(38-54), each IMU block (17 cols):
# [0]temp, [1..3]acc16, [4..6]acc6, [7..9]gyro, [10..12]mag, [13..16]orientation
def imu_block_cols_6axis(start_col_1idx: int):
    base0 = start_col_1idx - 1
    acc16 = [base0 + i for i in [1, 2, 3]]
    gyro = [base0 + i for i in [7, 8, 9]]
    return acc16 + gyro  # 6 channels: ax, ay, az, gx, gy, gz


def get_raw_cols():
    return {
        "imu_hand": imu_block_cols_6axis(4),
        "imu_chest": imu_block_cols_6axis(21),
        "imu_ankle": imu_block_cols_6axis(38),
        "heart_rate": [2],  # 3rd column (1-based)
    }


# =========================
# I/O & Resampling Utils
# =========================
def read_dat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 54:
        raise ValueError(f"{path.name}: expected 54 columns, got {df.shape[1]}")
    return df


def extract_subject_id(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def ensure_monotonic_unique(time_s: np.ndarray, x: np.ndarray):
    order = np.argsort(time_s)
    time_s = time_s[order]
    x = x[order]
    uniq_t, uniq_idx = np.unique(time_s, return_index=True)
    return uniq_t.astype(np.float64), x[uniq_idx].astype(np.float32)


def resample_to(time_s: np.ndarray, x: np.ndarray, target_hz: float):
    t, x = ensure_monotonic_unique(time_s, x)
    if len(t) == 0 or t[-1] <= t[0]:
        return np.array([]), np.empty((0, x.shape[1]), dtype=np.float32)
    new_t = np.arange(float(t[0]), float(t[-1]), 1.0 / target_hz, dtype=np.float64)
    # Per-channel interpolation with NaN handling
    x_filled = np.empty((len(t), x.shape[1]), dtype=np.float64)
    for d in range(x.shape[1]):
        s = pd.Series(x[:, d].astype(np.float64), index=t)
        s = s.interpolate(method="index", limit_direction="both").bfill().ffill()
        x_filled[:, d] = s.values
    out = np.empty((len(new_t), x.shape[1]), dtype=np.float32)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(new_t, t, x_filled[:, d]).astype(np.float32)
    return new_t, out


def window_segments(x: np.ndarray, win_len: int, hop_len: int) -> np.ndarray:
    T = x.shape[0]
    segs = []
    i = 0
    while i + win_len <= T:
        segs.append(x[i:i + win_len])
        i += hop_len
    return np.stack(segs) if segs else np.zeros((0, win_len, x.shape[1]), dtype=x.dtype)


def upsample_labels_to(new_t: np.ndarray, lab_t: np.ndarray, labels: np.ndarray):
    idx = np.searchsorted(lab_t, new_t, side='left')
    idx = np.clip(idx, 0, len(lab_t) - 1)
    return labels[idx]


def window_majority_labels(y_up: np.ndarray, win_len: int, hop_len: int, exclude={0}) -> np.ndarray:
    labs = []
    i = 0
    L = len(y_up)
    while i + win_len <= L:
        w = y_up[i:i + win_len]
        vals, cnt = np.unique(w, return_counts=True)
        maj = int(vals[np.argmax(cnt)])
        labs.append(-1 if maj in exclude else maj)
        i += hop_len
    return np.asarray(labs, dtype=np.int64)


# =========================
# Rolling / Feature helpers
# =========================
def _odd_k(k: int) -> int:
    """Ensure odd kernel size >= 3."""
    k = max(3, int(k))
    return k if (k % 2 == 1) else (k + 1)


def _compute_multiscale_windows(target_hz: float):
    # 0.25s, 0.5s, 1.0s in samples (rounded), forced to odd
    base_sec = [0.25, 0.5, 1.0]
    ks = [_odd_k(round(s * target_hz)) for s in base_sec]
    return ks  # e.g., [25, 51, 101] for 100 Hz


def _rolling_mean_1d(x1d: np.ndarray, k: int) -> np.ndarray:
    # reflect pad to achieve 'same' length for odd k
    pad = (k - 1) // 2
    xpad = np.pad(x1d, (pad, pad), mode='reflect')
    kernel = np.ones(k, dtype=np.float32) / k
    y = np.convolve(xpad, kernel, mode='valid').astype(np.float32)
    # length should equal len(x1d)
    if y.shape[0] != x1d.shape[0]:
        # fallback to 'same' if numerical oddities
        y = np.convolve(x1d, kernel, mode='same').astype(np.float32)
    return y


def _rolling_mean_multi(X: np.ndarray, k: int) -> np.ndarray:
    return np.stack([_rolling_mean_1d(X[:, d], k) for d in range(X.shape[1])], axis=1)


def _rolling_stats(X: np.ndarray, k: int):
    # X: (T, D)
    m = _rolling_mean_multi(X, k)
    m_sq = _rolling_mean_multi(X ** 2, k)
    var = np.clip(m_sq - m ** 2, a_min=0.0, a_max=None)
    std = np.sqrt(var, dtype=np.float32)
    rms = np.sqrt(m_sq, dtype=np.float32)
    return m, std, rms  # (T,D) each


def _diff_keep_len(X: np.ndarray):
    # first difference with same length (prefix with first row)
    d = np.diff(X, axis=0, prepend=X[0:1, :])
    return d.astype(np.float32)


def _central_slope_1d(x1d: np.ndarray, k: int) -> np.ndarray:
    # central difference slope over window half-width Δ=(k//2)
    Δ = max(1, k // 2)
    xp = np.pad(x1d, (Δ, Δ), mode='reflect')
    num = xp[2 * Δ:] - xp[:-2 * Δ]
    den = 2.0 * Δ
    y = (num / den).astype(np.float32)
    if y.shape[0] != x1d.shape[0]:
        # pad to match length if needed (rare)
        if y.shape[0] < x1d.shape[0]:
            y = np.pad(y, (0, x1d.shape[0] - y.shape[0]), mode='edge')
        else:
            y = y[:x1d.shape[0]]
    return y


def _central_slope_multi(X: np.ndarray, k: int) -> np.ndarray:
    return np.stack([_central_slope_1d(X[:, d], k) for d in range(X.shape[1])], axis=1)


# =========================
# Feature Engineering (No zero padding)
# =========================
def features64_from_imu_window(win_6ch: np.ndarray, ks: list[int]) -> np.ndarray:
    """
    IMU window: (T,6) -> (T,64) using multiscale stats & dynamics.
    Channels: [ax, ay, az, gx, gy, gz]
    ks: list of odd kernel sizes (e.g., [25, 51, 101])
    """
    x = win_6ch.astype(np.float32)
    assert x.shape[1] == 6, f"expected 6-ch IMU, got {x.shape}"
    feats = []

    # 1) per-channel rolling mean/std at 3 scales -> 2*3*6 = 36
    for k in ks:
        m, s, _ = _rolling_stats(x, k)
        feats += [m, s]

    # 2) accel RMS (3 scales) -> 3*3 = 9 => 45
    acc = x[:, :3]
    for k in ks:
        _, _, rms = _rolling_stats(acc, k)
        feats.append(rms)  # (T,3)

    # 3) magnitude (acc, gyro) mean/std at 3 scales -> 2*2*3 = 12 => 57
    gyro = x[:, 3:]
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    for sig in [acc_mag, gyro_mag]:
        for k in ks:
            m, s, _ = _rolling_stats(sig, k)  # (T,1) each
            feats += [m, s]

    # 4) first diff for 6 channels -> 6 => 63
    feats.append(_diff_keep_len(x))

    # 5) accel jerk magnitude rolling mean at the shortest scale -> 1 => 64
    jerk = _diff_keep_len(acc)              # (T,3)
    jerk_mag = np.linalg.norm(jerk, axis=1, keepdims=True)  # (T,1)
    feats.append(_rolling_mean_multi(jerk_mag, ks[0]))      # (T,1)

    Z = np.concatenate(feats, axis=1)
    assert Z.shape[1] == 64, f"IMU feature dim mismatch: {Z.shape}"
    return Z


def features64_from_hr_window(win_hr: np.ndarray, ks: list[int]) -> np.ndarray:
    """
    HR window: (T,1) -> (T,64) using lags, rolling stats, EWMA, diffs, slopes.
    ks: list of odd kernel sizes (e.g., [25, 51, 101])
    """
    x = win_hr.astype(np.float32)
    assert x.shape[1] == 1, f"expected 1-ch HR, got {x.shape}"
    T = x.shape[0]
    feats = []

    # 1) current + lags 1..30 -> 31
    feats.append(x)
    for lag in range(1, 31):
        v = np.roll(x, shift=lag, axis=0).copy()
        v[:lag, :] = x[0, :]  # boundary fix
        feats.append(v)

    # 2) rolling mean/std/rms at 3 scales -> 9 -> total 40
    for k in ks:
        m, s, r = _rolling_stats(x, k)
        feats += [m, s, r]

    # 3) EWMA/EWMSTD (alpha=0.1, 0.01) -> 4 -> total 44
    for alpha in [0.1, 0.01]:
        ewma = np.zeros_like(x)
        ewma[0] = x[0]
        for t in range(1, T):
            ewma[t] = alpha * x[t] + (1 - alpha) * ewma[t - 1]
        # EWM std approx: rolling std over EWMA with middle scale
        ewmstd = _rolling_stats(ewma, ks[1])[1]
        feats += [ewma, ewmstd]

    # 4) first & second difference -> 2 -> total 46
    d1 = _diff_keep_len(x)
    d2 = _diff_keep_len(d1)
    feats += [d1, d2]

    # 5) z-score (middle scale) + central slope for 3 scales -> 1 + 3 = 4 -> total 50
    m_mid, s_mid, _ = _rolling_stats(x, ks[1])
    z = (x - m_mid) / (s_mid + 1e-6)
    feats.append(z)
    for k in ks:
        feats.append(_central_slope_multi(x, k))

    # 6) additional lags 31..44 -> 14 -> total 64
    for lag in range(31, 45):
        v = np.roll(x, shift=lag, axis=0).copy()
        v[:lag, :] = x[0, :]
        feats.append(v)

    Z = np.concatenate(feats, axis=1)
    assert Z.shape[1] == 64, f"HR feature dim mismatch: {Z.shape}"
    return Z


# =========================
# Subject split
# =========================
def split_subjects(files, train_r, val_r, test_r):
    subs = sorted({extract_subject_id(p) for p in files})
    n = len(subs)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    # remainder -> test
    return {
        "train": set(subs[:n_train]),
        "val": set(subs[n_train:n_train + n_val]),
        "test": set(subs[n_train + n_val:]),
    }


# =========================
# Per-file processing
# =========================
def process_df(df: pd.DataFrame, target_hz=100.0, win_sec=5.0, hop_sec=2.5):
    """
    Returns:
      wins: dict modality -> windows array
            imu_*: (N, T, 6)
            heart_rate: (N, T, 1)
      y_win: (N,)
    or None if empty
    """
    time = df.iloc[:, 0].to_numpy(dtype=np.float64)
    y = df.iloc[:, 1].to_numpy(dtype=np.int64)
    cols = get_raw_cols()
    raw = {k: df.iloc[:, idx].to_numpy(dtype=np.float32) for k, idx in cols.items()}

    new_t, hand = resample_to(time, raw["imu_hand"], target_hz)
    if new_t.size == 0:
        return None
    _, chest = resample_to(time, raw["imu_chest"], target_hz)
    _, ankle = resample_to(time, raw["imu_ankle"], target_hz)
    _, hr = resample_to(time, raw["heart_rate"].reshape(-1, 1), target_hz)

    win_len = int(win_sec * target_hz)
    hop_len = int(hop_sec * target_hz)

    y_up = upsample_labels_to(new_t, time, y)
    y_win = window_majority_labels(y_up, win_len, hop_len, exclude={0})

    wins = {
        "imu_hand": window_segments(hand, win_len, hop_len),
        "imu_chest": window_segments(chest, win_len, hop_len),
        "imu_ankle": window_segments(ankle, win_len, hop_len),
        "heart_rate": window_segments(hr, win_len, hop_len),
    }

    valid = (y_win != -1)
    if not valid.any():
        return None

    for k in wins:
        wins[k] = wins[k][valid]
    y_win = y_win[valid]
    return wins, y_win


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to base.yaml")
    ap.add_argument("--raw_dir", type=str, required=True, help="path to PAMAP2 raw .dat files")
    ap.add_argument("--out_dir", type=str, required=True, help="output dir (…/processed)")
    ap.add_argument("--target_hz", type=float, default=100.0)
    ap.add_argument("--window_sec", type=float, default=5.0)
    ap.add_argument("--hop_sec", type=float, default=2.5)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = list(cfg.dataset.modalities)  # e.g., [imu_hand, imu_chest, imu_ankle, heart_rate]

    # Confirm target dims (should be 64 given the YAML & train.py defaults)
    target_dims = {m: get_target_dim_for_mod(cfg, m) for m in modalities}
    for m, d in target_dims.items():
        if d != 64:
            print(f"[WARN] modality '{m}' input_dim={d} (expected 64). "
                  f"This script produces 64-dim features per timestep.")

    files = sorted(list(raw_dir.glob("*.dat")))
    assert files, f"No .dat files in {raw_dir}"
    splits = split_subjects(files, cfg.dataset.train_split, cfg.dataset.val_split, cfg.dataset.test_split)

    # multi-scale window sizes for rolling stats
    ks = _compute_multiscale_windows(args.target_hz)

    buffers = {
        "train": {m: [] for m in modalities} | {"labels": [], "subjects": []},
        "val": {m: [] for m in modalities} | {"labels": [], "subjects": []},
        "test": {m: [] for m in modalities} | {"labels": [], "subjects": []},
    }

    for f in files:
        sid = extract_subject_id(f)
        split = "train" if sid in splits["train"] else "val" if sid in splits["val"] else "test"

        df = read_dat(f)
        res = process_df(df, target_hz=args.target_hz, win_sec=args.window_sec, hop_sec=args.hop_sec)
        if res is None:
            continue
        wins, y_win = res  # raw windows: imu_* (N,T,6), hr (N,T,1)

        # Convert each window to (T,64) features (no zero padding)
        for m in modalities:
            arr_list = []
            if m.startswith("imu_"):
                for w in wins[m]:
                    arr_list.append(features64_from_imu_window(w, ks))
            elif m == "heart_rate":
                for w in wins[m]:
                    arr_list.append(features64_from_hr_window(w, ks))
            else:
                # not expected for PAMAP2, keep as-is if encountered
                for w in wins[m]:
                    arr_list.append(w)
            stacked = np.stack(arr_list, axis=0) if arr_list else np.zeros((0, 1, 64), dtype=np.float32)
            buffers[split][m].append(stacked)

        buffers[split]["labels"].append(y_win)
        buffers[split]["subjects"].append(np.full(y_win.shape[0], sid, dtype=np.int32))

    # Build global label map (compact)
    all_labels = []
    for sp in ["train", "val", "test"]:
        if buffers[sp]["labels"]:
            all_labels.append(np.concatenate(buffers[sp]["labels"]))
    all_labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)
    uniq = np.sort(np.unique(all_labels)) if all_labels.size else np.array([], dtype=np.int64)
    label_map = {lab: i for i, lab in enumerate(uniq)}

    # Save per split
    for sp in ["train", "val", "test"]:
        sp_dir = out_dir / sp
        sp_dir.mkdir(parents=True, exist_ok=True)

        sizes = {}
        for m in modalities:
            arrs = buffers[sp][m]
            if arrs:
                X = np.concatenate(arrs, axis=0)  # (N, T, 64)
            else:
                X = np.zeros((0, 1, 64), dtype=np.float32)
            np.save(sp_dir / f"{m}.npy", X)
            sizes[m] = X.shape

        labels = np.concatenate(buffers[sp]["labels"]) if buffers[sp]["labels"] else np.zeros((0,), np.int64)
        labels = np.vectorize(label_map.get)(labels).astype(np.int64) if labels.size else labels
        subjects = np.concatenate(buffers[sp]["subjects"]) if buffers[sp]["subjects"] else np.zeros((0,), np.int32)

        np.save(sp_dir / "labels.npy", labels)
        np.save(sp_dir / "subjects.npy", subjects)

        print(f"[{sp}] sizes={sizes}, labels={labels.shape}, subjects={subjects.shape}")


if __name__ == "__main__":
    main()
