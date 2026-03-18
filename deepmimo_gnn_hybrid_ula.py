#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepMIMO + Hybrid AoA GNN + Grouped MUSIC sweep

Updated version with:
1) DeepMIMO scenario loading directly
2) Hybrid AoA GNN:
   - coarse angle-bin classification
   - residual regression inside bin
3) ULA-aware training loss
4) Proper ULA-aware evaluation:
   min(|pred-gt|, |pred+gt|)
5) Grouped covariance node features
6) GATv2 / SAGE support with residual connections + LayerNorm
7) MUSIC + CRB + timing + RMSE/MAE outputs

Example:
python deepmimo_gnn_grouped_hybrid_ula.py \
  --scenario asu_campus_3p5 \
  --candidate-subc 64 \
  --subc-start 0 \
  --n-select-list 20,30,40 \
  --group-list 5,10 \
  --run-all-methods \
  --methods correlation fisher doptimal \
  --bs-x 8 --bs-y 1 \
  --ue-x 1 --ue-y 1 \
  --epochs 100 \
  --batch-size 64 \
  --conv-type gatv2 \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.05 \
  --topk 2 \
  --trace-norm \
  --num-angle-bins 37 \
  --class-loss-weight 1.0 \
  --residual-loss-weight 1.0 \
  --ula-loss-weight 0.8 \
  --outdir results_deepmimo_grouped_hybrid_ula

Install:
pip install numpy pandas matplotlib scipy
pip install torch torch-geometric
pip install --pre deepmimo
"""

import os
import json
import time
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import SAGEConv, GATv2Conv, AttentionalAggregation


# ============================================================
# General utils
# ============================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_getattr(obj, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def parse_int_list(s: str):
    if s is None or len(s.strip()) == 0:
        return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def pretty_method_name(method: str) -> str:
    mapping = {
        "correlation": "Correlation",
        "fisher": "Fisher",
        "doptimal": "D-optimal",
        "contiguous": "Contiguous",
    }
    return mapping.get(method, method)


def manual_split_indices(n: int, test_size: float, val_size: float, seed: int = 42):
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0,1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise RuntimeError("Split sizes resulted in empty train/val/test split")

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ============================================================
# Angle utils
# ============================================================

def wrap_to_180(angle_deg):
    angle_deg = np.asarray(angle_deg)
    return (angle_deg + 180.0) % 360.0 - 180.0


def fold_aoa_to_ula_range(angle_deg):
    """
    Fold physical azimuth in [-180, 180] to ULA-observable AoA in [-90, 90]:
        theta_ula = arcsin(sin(theta_gt))
    """
    ang = np.asarray(angle_deg, dtype=np.float64)
    folded = np.rad2deg(np.arcsin(np.sin(np.deg2rad(ang))))
    return folded.astype(np.float32)


def angular_error_deg(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = (y_pred - y_true + 180.0) % 360.0 - 180.0
    return err


def mae_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(angular_error_deg(y_true, y_pred))))


def rmse_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = angular_error_deg(y_true, y_pred)
    return float(np.sqrt(np.mean(err ** 2)))


def ula_abs_error_deg(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Proper ULA ambiguity-aware error:
        min( |pred-gt|, |pred+gt| )
    with angle wrapping.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    e1 = np.abs((y_pred - y_true + 180.0) % 360.0 - 180.0)
    e2 = np.abs((y_pred + y_true + 180.0) % 360.0 - 180.0)
    return np.minimum(e1, e2)


def mae_ula_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(ula_abs_error_deg(y_true, y_pred)))


def rmse_ula_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = ula_abs_error_deg(y_true, y_pred)
    return float(np.sqrt(np.mean(err ** 2)))


# ============================================================
# Angle bins
# ============================================================

def make_angle_bins(num_bins: int, min_deg: float = -90.0, max_deg: float = 90.0):
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    centers = np.linspace(min_deg, max_deg, num_bins, dtype=np.float32)
    if num_bins > 1:
        bin_width = float(centers[1] - centers[0])
    else:
        bin_width = max_deg - min_deg
    return centers, bin_width


def angle_to_class_and_residual(angle_deg: float, centers: np.ndarray, bin_width: float):
    diffs = np.abs(centers - angle_deg)
    cls = int(np.argmin(diffs))
    center = float(centers[cls])
    residual_deg = float(angle_deg - center)
    residual_norm = residual_deg / max(bin_width / 2.0, 1e-6)
    residual_norm = float(np.clip(residual_norm, -1.0, 1.0))
    return cls, residual_norm


def class_and_residual_to_angle_deg(class_idx: np.ndarray, residual_norm: np.ndarray, centers: np.ndarray, bin_width: float):
    class_idx = np.asarray(class_idx, dtype=np.int64)
    residual_norm = np.asarray(residual_norm, dtype=np.float64)
    center_vals = centers[class_idx].astype(np.float64)
    residual_deg = residual_norm * (bin_width / 2.0)
    return (center_vals + residual_deg).astype(np.float32)


def soft_angle_from_logits_and_residual(
    cls_logits: torch.Tensor,
    residual_norm: torch.Tensor,
    centers: np.ndarray,
    bin_width: float,
    device: torch.device
):
    """
    Differentiable angle decode:
    expected class center + residual correction.
    """
    centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
    probs = torch.softmax(cls_logits, dim=1)
    center_soft = torch.sum(probs * centers_t.unsqueeze(0), dim=1)
    pred_angle = center_soft + residual_norm * (bin_width / 2.0)
    pred_angle = torch.clamp(pred_angle, -90.0, 90.0)
    return pred_angle


def ula_aware_angle_loss(pred_angle: torch.Tensor, target_angle: torch.Tensor):
    """
    ULA-aware robust loss:
        min( Huber(pred-gt), Huber(pred+gt) )
    """
    e1 = pred_angle - target_angle
    e2 = pred_angle + target_angle
    z1 = torch.zeros_like(e1)
    z2 = torch.zeros_like(e2)

    l1 = F.smooth_l1_loss(e1, z1, reduction="none")
    l2 = F.smooth_l1_loss(e2, z2, reduction="none")
    return torch.mean(torch.minimum(l1, l2))


# ============================================================
# Steering / MUSIC / CRB
# ============================================================

def steering_vector_ula(num_ant: int, angle_rad: float, d_over_lambda: float = 0.5) -> np.ndarray:
    n = np.arange(num_ant)
    return np.exp(-1j * 2 * np.pi * d_over_lambda * n * np.sin(angle_rad))[:, None]


def steering_derivative_ula(num_ant: int, angle_rad: float, d_over_lambda: float = 0.5) -> np.ndarray:
    n = np.arange(num_ant)
    a = np.exp(-1j * 2 * np.pi * d_over_lambda * n * np.sin(angle_rad))
    coeff = -1j * 2 * np.pi * d_over_lambda * n * np.cos(angle_rad)
    da = coeff * a
    return da[:, None]


def estimate_covariance_from_subcarriers(h_sample: np.ndarray) -> np.ndarray:
    return (h_sample @ h_sample.conj().T) / max(1, h_sample.shape[1])


def music_spectrum_ula(
    R: np.ndarray,
    num_sources: int = 1,
    angle_grid_deg: Optional[np.ndarray] = None,
    d_over_lambda: float = 0.5
):
    if angle_grid_deg is None:
        angle_grid_deg = np.linspace(-90, 90, 721)

    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    if num_sources >= R.shape[0]:
        raise ValueError("num_sources must be smaller than number of antennas")

    En = eigvecs[:, num_sources:]
    P = []

    for ang_deg in angle_grid_deg:
        a = steering_vector_ula(R.shape[0], np.deg2rad(ang_deg), d_over_lambda=d_over_lambda)
        denom = np.linalg.norm(En.conj().T @ a) ** 2
        P.append(1.0 / max(denom, 1e-12))

    return angle_grid_deg, np.asarray(P, dtype=np.float64)


def find_specified_peaks(phi_deg: np.ndarray, spectrum: np.ndarray, L: int = 1) -> np.ndarray:
    peaks, _ = signal.find_peaks(np.abs(spectrum))
    if len(peaks) == 0:
        return np.array([phi_deg[np.argmax(spectrum)]], dtype=np.float32)

    peak_vals = spectrum[peaks]
    order = np.argsort(peak_vals)[::-1]
    peaks = peaks[order[:L]]
    return phi_deg[peaks].astype(np.float32)


def run_music_single_sample_grouped(
    h_sample: np.ndarray,
    groups_sc: List[List[int]],
    num_sources: int = 1,
    angle_grid_deg: Optional[np.ndarray] = None
):
    if angle_grid_deg is None:
        angle_grid_deg = np.linspace(-90, 90, 721)

    group_spectra = []
    for g in groups_sc:
        hg = h_sample[:, g]
        Rg = estimate_covariance_from_subcarriers(hg)
        phi_deg, P = music_spectrum_ula(Rg, num_sources=num_sources, angle_grid_deg=angle_grid_deg)
        group_spectra.append(np.abs(P))

    Pagg = np.mean(np.stack(group_spectra, axis=0), axis=0)
    est = find_specified_peaks(phi_deg, Pagg, L=1)[0]
    return float(est), Pagg


def estimate_snr_proxy_from_cov(R: np.ndarray, L: int = 1, eps: float = 1e-12):
    vals = np.linalg.eigvalsh(R)
    vals = np.sort(np.real(vals))[::-1]
    M = len(vals)
    noise = float(np.mean(vals[L:])) if (M - L) > 0 else float(vals[-1])
    signal_pow = float(max(vals[0] - noise, 0.0))
    return signal_pow / (noise + eps)


def approx_grouped_crb_deg(
    h_sample: np.ndarray,
    theta_deg: float,
    groups_sc: List[List[int]],
    d_over_lambda: float = 0.5,
    num_sources: int = 1
) -> float:
    M = h_sample.shape[0]
    theta = np.deg2rad(theta_deg)

    a = steering_vector_ula(M, theta, d_over_lambda=d_over_lambda)
    da = steering_derivative_ula(M, theta, d_over_lambda=d_over_lambda)

    ah_a = (a.conj().T @ a).item()
    P_a = (a @ a.conj().T) / max(np.real(ah_a), 1e-12)
    P_perp = np.eye(M, dtype=np.complex128) - P_a
    deriv_term = np.real((da.conj().T @ P_perp @ da).item())
    deriv_term = max(float(deriv_term), 1e-12)

    fim_total = 0.0
    for g in groups_sc:
        hg = h_sample[:, g]
        Rg = estimate_covariance_from_subcarriers(hg)
        snr_g = estimate_snr_proxy_from_cov(Rg, L=num_sources)
        fim_total += 2.0 * max(len(g), 1) * max(snr_g, 0.0) * deriv_term

    fim_total = max(fim_total, 1e-12)
    crb_rad2 = 1.0 / fim_total
    crb_deg = float(np.sqrt(crb_rad2) * (180.0 / np.pi))
    return crb_deg


# ============================================================
# DeepMIMO loading
# ============================================================

def load_deepmimo_candidate_channels(
    scenario: str = "asu_campus_3p5",
    bs_shape: Tuple[int, int] = (8, 1),
    ue_shape: Tuple[int, int] = (1, 1),
    candidate_subc: int = 64,
    subc_start: int = 0,
    max_users: Optional[int] = None,
    los_mode: str = "all",
    fold_labels_to_ula: bool = True,
):
    try:
        import deepmimo as dm
    except Exception as e:
        raise RuntimeError("Could not import deepmimo. Install with: pip install --pre deepmimo") from e

    try:
        dataset = dm.load(scenario)
    except Exception as e:
        raise RuntimeError(f"dm.load('{scenario}') failed: {e}") from e

    aoa_az = np.asarray(safe_getattr(dataset, "aoa_az"))
    if aoa_az is None or aoa_az.ndim != 2:
        raise RuntimeError("dataset.aoa_az is missing or has unexpected shape")

    labels_deg_all_raw = aoa_az[:, 0].astype(np.float32)
    valid_mask = np.isfinite(labels_deg_all_raw)

    los_arr = safe_getattr(dataset, "los", None)
    if los_arr is not None:
        los_arr = np.asarray(los_arr).reshape(-1)

    if los_mode == "los" and los_arr is not None:
        valid_mask &= (los_arr == 1)
    elif los_mode == "nlos" and los_arr is not None:
        valid_mask &= (los_arr != 1)
    elif los_mode not in ("all", "los", "nlos"):
        raise ValueError("los_mode must be one of: all, los, nlos")

    idxs = np.where(valid_mask)[0]
    if len(idxs) == 0:
        raise RuntimeError("No valid users remain after filtering")

    if max_users is not None:
        idxs = idxs[:max_users]

    trimmed = False
    try:
        dataset = dataset.trim(idxs)
        trimmed = True
        print(f"[INFO] dataset.trim() succeeded. Users kept: {len(idxs)}")
    except Exception as e:
        print(f"[WARN] dataset.trim() unavailable/failed. Manual slicing will be used. Error: {e}")
        trimmed = False

    ch_params = dm.ChannelParameters()
    ch_params.bs_antenna.shape = [bs_shape[0], bs_shape[1]]
    ch_params.ue_antenna.shape = [ue_shape[0], ue_shape[1]]
    ch_params.freq_domain = True
    ch_params.ofdm.selected_subcarriers = list(range(subc_start, subc_start + candidate_subc))

    channels = None
    try:
        channels = dataset.compute_channels(ch_params)
    except Exception:
        channels = safe_getattr(dataset, "channel", None)
        if channels is None:
            channels = safe_getattr(dataset, "channels", None)

    if channels is None:
        channels = safe_getattr(dataset, "channel", None)
    if channels is None:
        channels = safe_getattr(dataset, "channels", None)
    if channels is None:
        raise RuntimeError("DeepMIMO channel tensor not found after compute_channels()")

    channels = np.asarray(channels)

    if not trimmed:
        labels_deg_raw = labels_deg_all_raw[idxs]
        channels = channels[idxs]
    else:
        labels_deg_raw = np.asarray(safe_getattr(dataset, "aoa_az"))[:, 0].astype(np.float32)

    if fold_labels_to_ula:
        labels_deg = fold_aoa_to_ula_range(labels_deg_raw)
    else:
        labels_deg = labels_deg_raw.copy()

    meta = {
        "scenario": scenario,
        "candidate_subc": int(candidate_subc),
        "subc_start": int(subc_start),
        "num_samples": int(len(labels_deg)),
        "channels_shape": list(channels.shape),
        "los_mode": los_mode,
        "trimmed": trimmed,
        "bs_shape": list(bs_shape),
        "ue_shape": list(ue_shape),
        "raw_label_min_deg": float(np.min(labels_deg_raw)),
        "raw_label_max_deg": float(np.max(labels_deg_raw)),
        "processed_label_min_deg": float(np.min(labels_deg)),
        "processed_label_max_deg": float(np.max(labels_deg)),
        "fold_labels_to_ula": bool(fold_labels_to_ula),
    }
    return channels, labels_deg, meta


# ============================================================
# Channel helpers
# ============================================================

def squeeze_channel_to_bs_sc(channels: np.ndarray) -> np.ndarray:
    if channels.ndim == 4:
        return channels[:, 0, :, :]
    elif channels.ndim == 3:
        return channels
    else:
        raise ValueError(f"Unsupported channel shape: {channels.shape}")


def normalize_complex_channel(h: np.ndarray) -> np.ndarray:
    power = np.sqrt(np.mean(np.abs(h) ** 2) + 1e-12)
    return h / power


def cov_to_feature(R: np.ndarray, trace_norm: bool = True) -> np.ndarray:
    if trace_norm:
        tr = np.trace(R)
        if np.abs(tr) > 1e-12:
            R = R / tr
    feat = np.concatenate([np.real(R).reshape(-1), np.imag(R).reshape(-1)], axis=0)
    return feat.astype(np.float32)


def feature_per_subcarrier(ch_bs_sc: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(ch_bs_sc), axis=1)


def compute_subcarrier_signatures(ch_bs_sc: np.ndarray, trace_norm: bool = True) -> np.ndarray:
    N, M, S = ch_bs_sc.shape
    sigs = []
    for s in range(S):
        feat_acc = None
        for i in range(N):
            hs = ch_bs_sc[i, :, s:s+1]
            R = estimate_covariance_from_subcarriers(hs)
            feat = cov_to_feature(R, trace_norm=trace_norm)
            if feat_acc is None:
                feat_acc = feat.astype(np.float64)
            else:
                feat_acc += feat.astype(np.float64)
        sigs.append((feat_acc / max(1, N)).astype(np.float32))
    return np.stack(sigs, axis=0)


# ============================================================
# Subcarrier selection
# ============================================================

def corr_similarity_matrix(X: np.ndarray, use_abs: bool = True) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    if use_abs:
        sim = np.abs(sim)
    np.fill_diagonal(sim, 0.0)
    return sim


def select_corr_centrality(sim: np.ndarray, n_select: int, top_p: int = 10):
    K = sim.shape[0]
    top_p = min(top_p, K - 1)
    scores = np.zeros(K, dtype=np.float32)
    for k in range(K):
        row = sim[k]
        scores[k] = float(np.mean(np.sort(row)[-top_p:])) if top_p > 0 else 0.0
    ranked = np.argsort(scores)[::-1]
    return ranked[:n_select].tolist(), scores


def rank_subcarriers_fisher(ch_bs_sc: np.ndarray, labels_deg: np.ndarray, n_bins: int = 18) -> np.ndarray:
    feats = feature_per_subcarrier(ch_bs_sc)
    S = feats.shape[1]

    bins = np.linspace(-90, 90, n_bins + 1)
    yb = np.digitize(labels_deg, bins) - 1
    yb = np.clip(yb, 0, n_bins - 1)

    scores = np.zeros(S, dtype=np.float64)
    for s in range(S):
        x = feats[:, s]
        mu = np.mean(x)
        num = 0.0
        den = 0.0

        for c in range(n_bins):
            xc = x[yb == c]
            if len(xc) == 0:
                continue
            muc = np.mean(xc)
            num += len(xc) * (muc - mu) ** 2
            den += np.sum((xc - muc) ** 2)

        scores[s] = num / max(den, 1e-12)

    return scores.astype(np.float32)


def dopt_greedy_select_from_signatures(
    sigs: np.ndarray,
    n_select: int,
    proj_dim: int = 32,
    eps: float = 1e-6,
    seed: int = 7
):
    K, F = sigs.shape
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((F, proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
    V = sigs @ W
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    selected = []
    A_inv = np.linalg.inv(eps * np.eye(proj_dim, dtype=np.float64))
    remaining = set(range(K))

    for _ in range(n_select):
        best_k = None
        best_gain = -1e18
        for k in remaining:
            v = V[k].astype(np.float64).reshape(-1, 1)
            gain = float(np.log1p((v.T @ A_inv @ v).item()))
            if gain > best_gain:
                best_gain = gain
                best_k = k

        v = V[best_k].astype(np.float64).reshape(-1, 1)
        denom = 1.0 + (v.T @ A_inv @ v).item()
        A_inv = A_inv - (A_inv @ v @ v.T @ A_inv) / denom

        selected.append(best_k)
        remaining.remove(best_k)

    return selected


def greedy_group_by_similarity(selected, sim: np.ndarray, group_size: int = 5):
    selected = list(dict.fromkeys(selected))
    if len(selected) % group_size != 0:
        raise ValueError(f"len(selected)={len(selected)} must be divisible by group_size={group_size}")

    sel_list = sorted(selected)
    unassigned = set(sel_list)

    anchor_score = {}
    for k in sel_list:
        others = [j for j in sel_list if j != k]
        anchor_score[k] = float(np.mean([sim[k, j] for j in others])) if others else 0.0

    groups = []
    while unassigned:
        anchor = max(unassigned, key=lambda kk: anchor_score.get(kk, 0.0))
        unassigned.remove(anchor)
        candidates = sorted(list(unassigned), key=lambda j: sim[anchor, j], reverse=True)
        take = candidates[:group_size - 1]
        for j in take:
            unassigned.remove(j)
        groups.append([anchor] + take)

    return groups


def select_subcarriers_and_groups(
    ch_train_bs_sc: np.ndarray,
    labels_train: np.ndarray,
    method: str,
    n_select: int,
    group_size: int,
    corr_top_p: int = 10,
    fisher_bins: int = 18,
    dopt_proj_dim: int = 32,
    dopt_eps: float = 1e-6,
    dopt_seed: int = 7,
    trace_norm: bool = True
):
    S = ch_train_bs_sc.shape[-1]
    if n_select > S:
        raise ValueError(f"n_select={n_select} cannot exceed available subcarriers S={S}")
    if n_select % group_size != 0:
        raise ValueError(f"n_select={n_select} must be divisible by group_size={group_size}")

    sigs = compute_subcarrier_signatures(ch_train_bs_sc, trace_norm=trace_norm)
    sim = corr_similarity_matrix(sigs, use_abs=True)

    if method == "correlation":
        selected_rel_idx, scores = select_corr_centrality(sim, n_select=n_select, top_p=corr_top_p)
    elif method == "fisher":
        scores = rank_subcarriers_fisher(ch_train_bs_sc, labels_train, n_bins=fisher_bins)
        ranked = np.argsort(scores)[::-1]
        selected_rel_idx = ranked[:n_select].tolist()
    elif method == "doptimal":
        selected_rel_idx = dopt_greedy_select_from_signatures(
            sigs=sigs,
            n_select=n_select,
            proj_dim=dopt_proj_dim,
            eps=dopt_eps,
            seed=dopt_seed
        )
        scores = None
    elif method == "contiguous":
        selected_rel_idx = list(range(n_select))
        scores = np.zeros(S, dtype=np.float32)
    else:
        raise ValueError("method must be one of: contiguous, correlation, fisher, doptimal")

    groups_sc_candidate = greedy_group_by_similarity(selected_rel_idx, sim, group_size=group_size)
    selected_rel_idx_sorted = np.array(sorted(selected_rel_idx), dtype=int)

    local_map = {sc: i for i, sc in enumerate(selected_rel_idx_sorted.tolist())}
    groups_sc_local = [[local_map[sc] for sc in g] for g in groups_sc_candidate]

    return selected_rel_idx_sorted, groups_sc_local, groups_sc_candidate, scores


# ============================================================
# Graph conversion
# ============================================================

def channel_to_graph_grouped(
    ch_selected: np.ndarray,
    groups_sc: List[List[int]],
    label_deg: float,
    centers: np.ndarray,
    bin_width: float,
    topk: int = 0
) -> Data:
    ch_selected = normalize_complex_channel(ch_selected)
    S_selected = ch_selected.shape[1]

    node_feats = []
    for g in groups_sc:
        hg = ch_selected[:, g]
        Rg = estimate_covariance_from_subcarriers(hg)

        tr = np.trace(Rg)
        if np.abs(tr) > 1e-12:
            Rg = Rg / tr

        feat_cov = np.concatenate(
            [np.real(Rg).reshape(-1), np.imag(Rg).reshape(-1)],
            axis=0
        ).astype(np.float32)

        center_sc = np.mean(g) / max(S_selected - 1, 1)
        width_sc = len(g) / max(S_selected, 1)

        feat_meta = np.array([center_sc, width_sc], dtype=np.float32)
        feat = np.concatenate([feat_cov, feat_meta], axis=0)
        node_feats.append(feat)

    x_nodes = np.stack(node_feats, axis=0)
    N = x_nodes.shape[0]

    edges = []
    for i in range(N - 1):
        edges += [(i, i + 1), (i + 1, i)]

    if topk and N > 1:
        Xn = x_nodes / (np.linalg.norm(x_nodes, axis=1, keepdims=True) + 1e-9)
        sim = Xn @ Xn.T
        for i in range(N):
            sim[i, i] = -1.0
            nn_idx = np.argsort(sim[i])[-topk:]
            for j in nn_idx:
                edges += [(i, int(j)), (int(j), i)]

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    class_idx, residual_norm = angle_to_class_and_residual(float(label_deg), centers, bin_width)

    return Data(
        x=torch.tensor(x_nodes, dtype=torch.float32),
        edge_index=edge_index,
        y_cls=torch.tensor(class_idx, dtype=torch.long),
        y_res=torch.tensor([residual_norm], dtype=torch.float32),
        y_angle=torch.tensor([label_deg], dtype=torch.float32),
    )


def build_graph_dataset_grouped(
    ch_bs_sc: np.ndarray,
    labels_deg: np.ndarray,
    groups_sc: List[List[int]],
    centers: np.ndarray,
    bin_width: float,
    topk: int = 0
):
    dataset = []
    for i in range(len(labels_deg)):
        dataset.append(
            channel_to_graph_grouped(
                ch_selected=ch_bs_sc[i],
                groups_sc=groups_sc,
                label_deg=float(labels_deg[i]),
                centers=centers,
                bin_width=bin_width,
                topk=topk
            )
        )
    return dataset


# ============================================================
# GNN
# ============================================================

class HybridAoAGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_bins: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        conv_type: str = "gatv2",
        dropout: float = 0.05
    ):
        super().__init__()
        self.dropout = dropout
        self.conv_type = conv_type.lower()
        self.num_bins = num_bins

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if self.conv_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif self.conv_type == "gatv2":
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False))
            else:
                raise ValueError("conv_type must be 'sage' or 'gatv2'")
            self.norms.append(nn.LayerNorm(hidden_dim))

        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.class_head = nn.Linear(hidden_dim, num_bins)
        self.res_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + x_res
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        z = self.shared_head(x)

        cls_logits = self.class_head(z)
        residual_norm = torch.tanh(self.res_head(z)).squeeze(-1)
        return cls_logits, residual_norm


# ============================================================
# Train / eval
# ============================================================

def decode_predictions(cls_logits: torch.Tensor, residual_norm: torch.Tensor, centers: np.ndarray, bin_width: float):
    pred_class = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()
    pred_res = residual_norm.detach().cpu().numpy()
    pred_angle = class_and_residual_to_angle_deg(pred_class, pred_res, centers, bin_width)
    return pred_class, pred_res, pred_angle


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    centers,
    bin_width,
    class_loss_weight=1.0,
    residual_loss_weight=1.0,
    ula_loss_weight=0.8
):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        cls_logits, residual_norm = model(batch)
        target_cls = batch.y_cls.view(-1)
        target_res = batch.y_res.view(-1)
        target_angle = batch.y_angle.view(-1)

        loss_cls = F.cross_entropy(cls_logits, target_cls)
        loss_res = F.smooth_l1_loss(residual_norm, target_res)

        pred_angle_soft = soft_angle_from_logits_and_residual(
            cls_logits, residual_norm, centers, bin_width, device
        )
        loss_ula = ula_aware_angle_loss(pred_angle_soft, target_angle)

        loss = (
            class_loss_weight * loss_cls
            + residual_loss_weight * loss_res
            + ula_loss_weight * loss_ula
        )

        loss.backward()
        optimizer.step()

        n = target_cls.numel()
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_gnn(
    model,
    loader,
    device,
    centers,
    bin_width,
    class_loss_weight=1.0,
    residual_loss_weight=1.0,
    ula_loss_weight=0.8
):
    model.eval()

    all_true_angle = []
    all_true_cls = []
    all_pred_cls = []
    all_pred_angle = []

    total_time = 0.0
    total_n = 0
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        t0 = time.perf_counter()
        cls_logits, residual_norm = model(batch)
        t1 = time.perf_counter()

        target_cls = batch.y_cls.view(-1)
        target_res = batch.y_res.view(-1)
        target_angle = batch.y_angle.view(-1)

        loss_cls = F.cross_entropy(cls_logits, target_cls)
        loss_res = F.smooth_l1_loss(residual_norm, target_res)

        pred_angle_soft = soft_angle_from_logits_and_residual(
            cls_logits, residual_norm, centers, bin_width, device
        )
        loss_ula = ula_aware_angle_loss(pred_angle_soft, target_angle)

        loss = (
            class_loss_weight * loss_cls
            + residual_loss_weight * loss_res
            + ula_loss_weight * loss_ula
        )

        pred_class, pred_res, pred_angle = decode_predictions(
            cls_logits, residual_norm, centers, bin_width
        )

        all_true_angle.append(target_angle.cpu().numpy())
        all_true_cls.append(target_cls.cpu().numpy())
        all_pred_cls.append(pred_class)
        all_pred_angle.append(pred_angle)

        n = target_cls.numel()
        total_time += (t1 - t0)
        total_n += n
        total_loss += float(loss.item()) * n

    y_true_angle = np.concatenate(all_true_angle)
    y_true_cls = np.concatenate(all_true_cls)
    y_pred_cls = np.concatenate(all_pred_cls)
    y_pred_angle = np.concatenate(all_pred_angle)

    class_acc = float(np.mean(y_true_cls == y_pred_cls))

    return y_true_angle, y_pred_angle, {
        "loss": total_loss / max(total_n, 1),
        "class_acc": class_acc,
        "mae_signed": mae_deg(y_true_angle, y_pred_angle),
        "rmse_signed": rmse_deg(y_true_angle, y_pred_angle),
        "mae_ula": mae_ula_deg(y_true_angle, y_pred_angle),
        "rmse_ula": rmse_ula_deg(y_true_angle, y_pred_angle),
        "infer_time_total_sec": total_time,
        "infer_time_per_sample_sec": total_time / max(total_n, 1),
    }


def run_music_dataset_grouped(
    ch_bs_sc: np.ndarray,
    labels_deg: np.ndarray,
    groups_sc: List[List[int]],
    angle_grid_deg=None
):
    preds = []
    total_time = 0.0
    crb_deg_all = []
    spectra_all = []

    for i in range(ch_bs_sc.shape[0]):
        h = ch_bs_sc[i]

        t0 = time.perf_counter()
        pred, Pagg = run_music_single_sample_grouped(
            h_sample=h,
            groups_sc=groups_sc,
            num_sources=1,
            angle_grid_deg=angle_grid_deg
        )
        t1 = time.perf_counter()

        preds.append(pred)
        total_time += (t1 - t0)
        spectra_all.append(Pagg)

        try:
            crb_deg = approx_grouped_crb_deg(
                h_sample=h,
                theta_deg=float(labels_deg[i]),
                groups_sc=groups_sc,
                d_over_lambda=0.5,
                num_sources=1
            )
        except Exception:
            crb_deg = np.nan

        crb_deg_all.append(crb_deg)

    preds = np.asarray(preds, dtype=np.float32)
    crb_deg_all = np.asarray(crb_deg_all, dtype=np.float64)
    avg_crb_deg = float(np.nanmean(crb_deg_all)) if len(crb_deg_all) > 0 else np.nan

    return preds, spectra_all, crb_deg_all, {
        "mae_signed": mae_deg(labels_deg, preds),
        "rmse_signed": rmse_deg(labels_deg, preds),
        "mae_ula": mae_ula_deg(labels_deg, preds),
        "rmse_ula": rmse_ula_deg(labels_deg, preds),
        "infer_time_total_sec": total_time,
        "infer_time_per_sample_sec": total_time / max(len(preds), 1),
        "avg_crb_deg": avg_crb_deg,
    }


# ============================================================
# Plotting
# ============================================================

def plot_training_curve(train_losses, val_mae_signed, val_mae_ula, val_class_acc, outpath):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_mae_signed, label="Val MAE Signed")
    plt.plot(val_mae_ula, label="Val MAE ULA")
    plt.plot(val_class_acc, label="Val Class Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_scatter(y_true, y_pred, title, outpath):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Ground Truth AoA (deg)")
    plt.ylabel("Predicted AoA (deg)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_scatter_abs(y_true, y_pred, title, outpath):
    y_true = np.abs(y_true)
    y_pred = np.abs(y_pred)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("|Ground Truth AoA| (deg)")
    plt.ylabel("|Predicted AoA| (deg)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_time_comparison(gnn_ms, music_ms, outpath):
    plt.figure(figsize=(6, 5))
    plt.bar(["GNN", "MUSIC"], [gnn_ms, music_ms])
    plt.ylabel("Time per sample (ms)")
    plt.title("Inference Time Comparison")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# Single run
# ============================================================

def run_one_config(
    ch_bs_sc: np.ndarray,
    labels_deg: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    experiment_root: str,
    selection_method: str,
    n_select: int,
    group_size: int,
    angle_grid_deg: np.ndarray,
    centers: np.ndarray,
    bin_width: float
):
    S = ch_bs_sc.shape[-1]
    if n_select > S:
        raise ValueError(f"n_select={n_select} cannot exceed S={S}")
    if n_select % group_size != 0:
        raise ValueError(f"n_select={n_select} must be divisible by group_size={group_size}")

    run_dir = os.path.join(
        experiment_root,
        f"method_{selection_method}",
        f"sel{n_select}",
        f"group_{group_size}"
    )
    data_dir = os.path.join(run_dir, "data")
    gnn_figure_dir = os.path.join(run_dir, "gnn_figure")
    gnn_data_dir = os.path.join(run_dir, "gnn_data")
    ensure_dir(run_dir)
    ensure_dir(data_dir)
    ensure_dir(gnn_figure_dir)
    ensure_dir(gnn_data_dir)

    selected_rel_idx, groups_sc, groups_sc_candidate, scores = select_subcarriers_and_groups(
        ch_train_bs_sc=ch_bs_sc[train_idx],
        labels_train=labels_deg[train_idx],
        method=selection_method,
        n_select=n_select,
        group_size=group_size,
        corr_top_p=args.corr_top_p,
        fisher_bins=args.fisher_bins,
        dopt_proj_dim=args.dopt_proj_dim,
        dopt_eps=args.dopt_eps,
        dopt_seed=args.dopt_seed,
        trace_norm=args.trace_norm
    )

    groups_sc = sorted(groups_sc, key=lambda g: np.mean(g))
    selected_abs_idx = selected_rel_idx + args.subc_start
    groups_sc_absolute = [[int(sc + args.subc_start) for sc in g] for g in groups_sc_candidate]

    selection_meta = {
        "selection_method": selection_method,
        "n_select": n_select,
        "group_size": group_size,
        "num_groups": len(groups_sc),
        "selected_relative_subcarriers": selected_rel_idx.tolist(),
        "selected_absolute_subcarriers": selected_abs_idx.tolist(),
        "groups_sc_candidate_window": groups_sc_candidate,
        "groups_sc_local_in_selected_tensor": groups_sc,
        "groups_sc_absolute": groups_sc_absolute,
    }
    save_json(selection_meta, os.path.join(gnn_data_dir, "selection_groups.json"))

    ch_selected = ch_bs_sc[:, :, selected_rel_idx]

    all_graphs = build_graph_dataset_grouped(
        ch_bs_sc=ch_selected,
        labels_deg=labels_deg,
        groups_sc=groups_sc,
        centers=centers,
        bin_width=bin_width,
        topk=args.topk
    )

    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs = [all_graphs[i] for i in val_idx]
    test_graphs = [all_graphs[i] for i in test_idx]

    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    test_ch_selected = ch_selected[test_idx]
    test_labels = labels_deg[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = train_graphs[0].x.shape[1]

    model = HybridAoAGNN(
        in_dim=in_dim,
        num_bins=len(centers),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_mae_ula = float("inf")
    best_state = None
    train_losses = []
    val_maes_signed = []
    val_maes_ula = []
    val_class_accs = []

    train_t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, centers, bin_width,
            class_loss_weight=args.class_loss_weight,
            residual_loss_weight=args.residual_loss_weight,
            ula_loss_weight=args.ula_loss_weight
        )
        _, _, val_metrics = evaluate_gnn(
            model, val_loader, device, centers, bin_width,
            class_loss_weight=args.class_loss_weight,
            residual_loss_weight=args.residual_loss_weight,
            ula_loss_weight=args.ula_loss_weight
        )

        train_losses.append(train_loss)
        val_maes_signed.append(val_metrics["mae_signed"])
        val_maes_ula.append(val_metrics["mae_ula"])
        val_class_accs.append(val_metrics["class_acc"])

        if val_metrics["mae_ula"] < best_val_mae_ula:
            best_val_mae_ula = val_metrics["mae_ula"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{selection_method} | n_select={n_select} | g={group_size} | Epoch {epoch:03d}] "
            f"Train Loss={train_loss:.6f} | "
            f"Val Class Acc={val_metrics['class_acc']:.4f} | "
            f"Val MAE Signed={val_metrics['mae_signed']:.4f} | "
            f"Val RMSE Signed={val_metrics['rmse_signed']:.4f} | "
            f"Val MAE ULA={val_metrics['mae_ula']:.4f} | "
            f"Val RMSE ULA={val_metrics['rmse_ula']:.4f}"
        )

    train_t1 = time.perf_counter()
    total_training_time_sec = train_t1 - train_t0

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "conv_type": args.conv_type,
            "dropout": args.dropout,
            "num_angle_bins": len(centers),
            "angle_bin_centers": centers.tolist(),
            "bin_width": bin_width,
            "selection_method": selection_method,
            "n_select": n_select,
            "group_size": group_size,
            "selected_relative_subcarriers": selected_rel_idx.tolist(),
            "selected_absolute_subcarriers": selected_abs_idx.tolist(),
            "groups_sc_relative_to_selected": groups_sc,
        },
        os.path.join(gnn_data_dir, "gnn_model_state.pt")
    )

    y_true_gnn, y_pred_gnn, gnn_metrics = evaluate_gnn(
        model, test_loader, device, centers, bin_width,
        class_loss_weight=args.class_loss_weight,
        residual_loss_weight=args.residual_loss_weight,
        ula_loss_weight=args.ula_loss_weight
    )
    np.save(os.path.join(gnn_data_dir, "gnn_predictions.npy"), y_pred_gnn)

    y_pred_music, spectra_all, crb_deg_all, music_metrics = run_music_dataset_grouped(
        ch_bs_sc=test_ch_selected,
        labels_deg=test_labels,
        groups_sc=groups_sc,
        angle_grid_deg=angle_grid_deg
    )

    rows = []
    gnn_err_signed = angular_error_deg(test_labels, y_pred_gnn)
    music_err_signed = angular_error_deg(test_labels, y_pred_music)
    gnn_err_ula = ula_abs_error_deg(test_labels, y_pred_gnn)
    music_err_ula = ula_abs_error_deg(test_labels, y_pred_music)

    for i in range(len(test_labels)):
        rows.append({
            "sample": int(i),
            "gt_doa": float(test_labels[i]),
            "gnn_doa": float(y_pred_gnn[i]),
            "music_doa": float(y_pred_music[i]),

            "gnn_abs_error_signed": float(abs(gnn_err_signed[i])),
            "music_abs_error_signed": float(abs(music_err_signed[i])),
            "gnn_sq_error_signed": float(gnn_err_signed[i] ** 2),
            "music_sq_error_signed": float(music_err_signed[i] ** 2),

            "gnn_abs_error_ula": float(gnn_err_ula[i]),
            "music_abs_error_ula": float(music_err_ula[i]),
            "gnn_sq_error_ula": float(gnn_err_ula[i] ** 2),
            "music_sq_error_ula": float(music_err_ula[i] ** 2),

            "gnn_train_time_total_sec": float(total_training_time_sec),
            "gnn_infer_time_sec": float(gnn_metrics["infer_time_per_sample_sec"]),
            "music_infer_time_sec": float(music_metrics["infer_time_per_sample_sec"]),
            "crb_deg": float(crb_deg_all[i]) if np.isfinite(crb_deg_all[i]) else np.nan,
            "selection_method": selection_method,
            "n_select": int(n_select),
            "group_size": int(group_size),
            "num_groups": int(len(groups_sc)),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    for i in range(min(len(test_labels), args.max_plot_samples)):
        Pagg = spectra_all[i]
        gt = test_labels[i]
        gnn = y_pred_gnn[i]
        music = y_pred_music[i]

        plt.figure(figsize=(7, 5))
        plt.semilogy(angle_grid_deg, Pagg / (np.max(Pagg) + 1e-12), label="Grouped MUSIC")
        plt.axvline(gt, linestyle="-", linewidth=2, label=f"GT={gt:.1f}°")
        plt.axvline(music, linestyle="--", linewidth=2, label=f"MUSIC={music:.1f}°")
        plt.axvline(gnn, linestyle=":", linewidth=2.5, label=f"GNN={gnn:.1f}°")
        plt.xlabel("Angle (degree)")
        plt.ylabel("Spectrum (normalized)")
        plt.title(
            f"Test sample {i} | {selection_method} | sel={n_select} | g={group_size}\n"
            f"MUSIC ULA err={gnn_err_ula[i] if False else abs(ula_abs_error_deg(np.array([gt]), np.array([music]))[0]):.2f}°, "
            f"GNN ULA err={abs(ula_abs_error_deg(np.array([gt]), np.array([gnn]))[0]):.2f}°"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(gnn_figure_dir, f"sample_{i}_gnn_music_comparison.png"), dpi=200)
        plt.close()

    plot_training_curve(
        train_losses,
        val_maes_signed,
        val_maes_ula,
        val_class_accs,
        os.path.join(run_dir, "training_curve.png")
    )

    plot_scatter(
        test_labels,
        y_pred_gnn,
        f"GNN Signed: GT vs Predicted ({selection_method}, sel={n_select}, g={group_size})",
        os.path.join(run_dir, "scatter_gnn_signed.png")
    )

    plot_scatter(
        test_labels,
        y_pred_music,
        f"MUSIC Signed: GT vs Predicted ({selection_method}, sel={n_select}, g={group_size})",
        os.path.join(run_dir, "scatter_music_signed.png")
    )

    plot_scatter_abs(
        test_labels,
        y_pred_gnn,
        f"GNN ULA: |GT| vs |Pred| ({selection_method}, sel={n_select}, g={group_size})",
        os.path.join(run_dir, "scatter_gnn_ula.png")
    )

    plot_scatter_abs(
        test_labels,
        y_pred_music,
        f"MUSIC ULA: |GT| vs |Pred| ({selection_method}, sel={n_select}, g={group_size})",
        os.path.join(run_dir, "scatter_music_ula.png")
    )

    plot_time_comparison(
        gnn_metrics["infer_time_per_sample_sec"] * 1000.0,
        music_metrics["infer_time_per_sample_sec"] * 1000.0,
        os.path.join(run_dir, "time_comparison.png")
    )

    num_plot = min(len(test_labels), args.max_doa_curve_samples)
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels[:num_plot], label="GT")
    plt.plot(y_pred_music[:num_plot], label="MUSIC")
    plt.plot(y_pred_gnn[:num_plot], label="GNN")
    plt.xlabel("Test sample")
    plt.ylabel("AoA (deg)")
    plt.title(
        f"DoA comparison ({selection_method}, sel={n_select}, g={group_size}) "
        f"[first {num_plot} test samples]"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "DoA_GT_vs_GNN_vs_MUSIC.png"), dpi=200)
    plt.close()

    run_summary = {
        "selection_method": selection_method,
        "n_select": n_select,
        "group_size": group_size,
        "num_groups": len(groups_sc),
        "selected_relative_subcarriers": selected_rel_idx.tolist(),
        "selected_absolute_subcarriers": selected_abs_idx.tolist(),

        "gnn_training_time_total_sec": total_training_time_sec,
        "gnn_class_acc": gnn_metrics["class_acc"],
        "gnn_mae_signed": gnn_metrics["mae_signed"],
        "gnn_rmse_signed": gnn_metrics["rmse_signed"],
        "gnn_mae_ula": gnn_metrics["mae_ula"],
        "gnn_rmse_ula": gnn_metrics["rmse_ula"],

        "music_mae_signed": music_metrics["mae_signed"],
        "music_rmse_signed": music_metrics["rmse_signed"],
        "music_mae_ula": music_metrics["mae_ula"],
        "music_rmse_ula": music_metrics["rmse_ula"],

        "gnn_time_per_sample_ms": gnn_metrics["infer_time_per_sample_sec"] * 1000.0,
        "music_time_per_sample_ms": music_metrics["infer_time_per_sample_sec"] * 1000.0,
        "avg_crb_deg": music_metrics["avg_crb_deg"],
    }
    save_json(run_summary, os.path.join(run_dir, "summary.json"))

    print(
        f"[DONE] {selection_method} | n_select={n_select} | g={group_size} | "
        f"GNN Class Acc={gnn_metrics['class_acc']:.4f} | "
        f"GNN MAE signed={gnn_metrics['mae_signed']:.4f} | "
        f"GNN MAE ULA={gnn_metrics['mae_ula']:.4f} | "
        f"MUSIC MAE signed={music_metrics['mae_signed']:.4f} | "
        f"MUSIC MAE ULA={music_metrics['mae_ula']:.4f} | "
        f"GNN train={total_training_time_sec:.4f} s | "
        f"GNN infer={gnn_metrics['infer_time_per_sample_sec']*1000:.4f} ms | "
        f"MUSIC infer={music_metrics['infer_time_per_sample_sec']*1000:.4f} ms | "
        f"CRB={music_metrics['avg_crb_deg']:.4f}"
    )

    return run_dir, {
        "selection_method": selection_method,
        "n_select": n_select,
        "group_size": group_size,
        "num_groups": len(groups_sc),

        "gnn_training_time_total_sec": total_training_time_sec,
        "gnn_class_acc": gnn_metrics["class_acc"],
        "gnn_mae_signed": gnn_metrics["mae_signed"],
        "gnn_rmse_signed": gnn_metrics["rmse_signed"],
        "gnn_mae_ula": gnn_metrics["mae_ula"],
        "gnn_rmse_ula": gnn_metrics["rmse_ula"],
        "gnn_time_per_sample": gnn_metrics["infer_time_per_sample_sec"],

        "music_mae_signed": music_metrics["mae_signed"],
        "music_rmse_signed": music_metrics["rmse_signed"],
        "music_mae_ula": music_metrics["mae_ula"],
        "music_rmse_ula": music_metrics["rmse_ula"],
        "music_time_per_sample": music_metrics["infer_time_per_sample_sec"],

        "avg_crb_deg": music_metrics["avg_crb_deg"],
        "selected_subcarriers": selected_abs_idx.tolist(),
        "run_dir": run_dir,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", type=str, default="asu_campus_3p5")
    parser.add_argument("--bs-x", type=int, default=8)
    parser.add_argument("--bs-y", type=int, default=1)
    parser.add_argument("--ue-x", type=int, default=1)
    parser.add_argument("--ue-y", type=int, default=1)

    parser.add_argument("--candidate-subc", type=int, default=64)
    parser.add_argument("--subc-start", type=int, default=0)
    parser.add_argument("--group-list", type=str, default="5,10")

    parser.add_argument("--n-select-list", type=str, default="20")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["correlation", "fisher", "doptimal"],
        choices=["contiguous", "correlation", "fisher", "doptimal"]
    )
    parser.add_argument("--run-all-methods", action="store_true")

    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--los-mode", type=str, default="all", choices=["all", "los", "nlos"])

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--corr-top-p", type=int, default=10)
    parser.add_argument("--fisher-bins", type=int, default=18)
    parser.add_argument("--dopt-proj-dim", type=int, default=32)
    parser.add_argument("--dopt-eps", type=float, default=1e-6)
    parser.add_argument("--dopt-seed", type=int, default=7)

    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--trace-norm", action="store_true")
    parser.add_argument("--conv-type", type=str, default="gatv2", choices=["sage", "gatv2"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--num-angle-bins", type=int, default=37)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--residual-loss-weight", type=float, default=1.0)
    parser.add_argument("--ula-loss-weight", type=float, default=0.8)

    parser.add_argument("--max-plot-samples", type=int, default=30)
    parser.add_argument("--max-doa-curve-samples", type=int, default=300)

    parser.add_argument("--outdir", type=str, default="results_deepmimo_grouped_hybrid_ula")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    n_select_list = parse_int_list(args.n_select_list)
    if len(n_select_list) == 0:
        raise ValueError("n-select-list is empty. Example: --n-select-list 20,30,40")

    methods = args.methods if args.run_all_methods else [args.methods[0]]
    group_values = parse_int_list(args.group_list)

    centers, bin_width = make_angle_bins(args.num_angle_bins, -90.0, 90.0)

    channels_candidate, labels_deg, meta = load_deepmimo_candidate_channels(
        scenario=args.scenario,
        bs_shape=(args.bs_x, args.bs_y),
        ue_shape=(args.ue_x, args.ue_y),
        candidate_subc=args.candidate_subc,
        subc_start=args.subc_start,
        max_users=args.max_users,
        los_mode=args.los_mode,
        fold_labels_to_ula=True,
    )

    print("[INFO] Candidate data loaded")
    print(json.dumps(meta, indent=2))
    print(f"[INFO] Angle bins: num_bins={len(centers)} | bin_width={bin_width:.4f} deg")

    ch_bs_sc = squeeze_channel_to_bs_sc(channels_candidate)
    N, M, S = ch_bs_sc.shape

    train_idx, val_idx, test_idx = manual_split_indices(
        n=N,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed
    )

    print(f"[INFO] Dataset split: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

    summary_rows = []
    final_table_rows = []
    skipped_rows = []

    angle_grid_deg = np.linspace(-90, 90, 721)

    for method in methods:
        for nsel in n_select_list:
            for gsize in group_values:
                if nsel > S:
                    print(f"[SKIP] method={method} | n_select={nsel} exceeds available S={S}")
                    skipped_rows.append({
                        "selection_method": method,
                        "n_select": nsel,
                        "group_size": gsize,
                        "reason": f"n_select > available_subcarriers({S})"
                    })
                    continue

                if nsel % gsize != 0:
                    print(f"[SKIP] method={method} | n_select={nsel} | group_size={gsize} (not divisible)")
                    skipped_rows.append({
                        "selection_method": method,
                        "n_select": nsel,
                        "group_size": gsize,
                        "reason": "n_select not divisible by group_size"
                    })
                    continue

                print("\n" + "=" * 100)
                print(f"[RUN] method={method} | n_select={nsel} | group_size={gsize}")
                print("=" * 100)

                run_dir, summ = run_one_config(
                    ch_bs_sc=ch_bs_sc,
                    labels_deg=labels_deg,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    args=args,
                    experiment_root=args.outdir,
                    selection_method=method,
                    n_select=nsel,
                    group_size=gsize,
                    angle_grid_deg=angle_grid_deg,
                    centers=centers,
                    bin_width=bin_width
                )

                summary_rows.append(summ)

                final_table_rows.append({
                    "Method": pretty_method_name(method),
                    "nselect": nsel,
                    "GroupSize": gsize,
                    "NumGroups": summ["num_groups"],

                    "GNN Train (s)": summ["gnn_training_time_total_sec"],
                    "GNN Class Acc": summ["gnn_class_acc"],
                    "GNN (ms)": summ["gnn_time_per_sample"] * 1000.0,
                    "MUSIC (ms)": summ["music_time_per_sample"] * 1000.0,

                    "GNN MAE Signed": summ["gnn_mae_signed"],
                    "GNN RMSE Signed": summ["gnn_rmse_signed"],
                    "GNN MAE ULA": summ["gnn_mae_ula"],
                    "GNN RMSE ULA": summ["gnn_rmse_ula"],

                    "MUSIC MAE Signed": summ["music_mae_signed"],
                    "MUSIC RMSE Signed": summ["music_rmse_signed"],
                    "MUSIC MAE ULA": summ["music_mae_ula"],
                    "MUSIC RMSE ULA": summ["music_rmse_ula"],

                    "CRB": summ["avg_crb_deg"],
                    "Selected Subcarriers": ",".join(map(str, summ["selected_subcarriers"]))
                })

    if len(summary_rows) == 0:
        raise RuntimeError("No valid runs were executed. Check n-select-list and group-list.")

    save_json(meta, os.path.join(args.outdir, "dataset_meta.json"))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)

    skipped_df = pd.DataFrame(skipped_rows)
    if len(skipped_df) > 0:
        skipped_df.to_csv(os.path.join(args.outdir, "skipped_combinations.csv"), index=False)

    final_df = pd.DataFrame(final_table_rows)
    method_order = {
        "Correlation": 0,
        "Fisher": 1,
        "D-optimal": 2,
        "Contiguous": 3
    }
    final_df["method_order"] = final_df["Method"].map(lambda x: method_order.get(x, 999))
    final_df = final_df.sort_values(["nselect", "GroupSize", "method_order"]).drop(columns=["method_order"])

    final_csv = os.path.join(args.outdir, "all_methods_results.csv")
    final_txt = os.path.join(args.outdir, "all_methods_results.txt")
    final_df.to_csv(final_csv, index=False)
    with open(final_txt, "w") as f:
        f.write(final_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    for method in methods:
        sdf = summary_df[summary_df["selection_method"] == method]
        if len(sdf) == 0:
            continue
        for gsize in sorted(sdf["group_size"].unique()):
            sdfg = sdf[sdf["group_size"] == gsize].sort_values("n_select")
            plt.plot(sdfg["n_select"], sdfg["gnn_mae_signed"], label=f"{method} g={gsize} GNN signed")
            plt.plot(sdfg["n_select"], sdfg["music_mae_signed"], label=f"{method} g={gsize} MUSIC signed")
    plt.xlabel("n_select")
    plt.ylabel("MAE (deg)")
    plt.title("Signed MAE vs n_select")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "comparison_mae_signed.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for method in methods:
        sdf = summary_df[summary_df["selection_method"] == method]
        if len(sdf) == 0:
            continue
        for gsize in sorted(sdf["group_size"].unique()):
            sdfg = sdf[sdf["group_size"] == gsize].sort_values("n_select")
            plt.plot(sdfg["n_select"], sdfg["gnn_mae_ula"], label=f"{method} g={gsize} GNN ULA")
            plt.plot(sdfg["n_select"], sdfg["music_mae_ula"], label=f"{method} g={gsize} MUSIC ULA")
    plt.xlabel("n_select")
    plt.ylabel("MAE (deg)")
    plt.title("ULA-aware MAE vs n_select")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "comparison_mae_ula.png"), dpi=200)
    plt.close()

    print("\n" + "=" * 120)
    print("[INFO] FINAL RESULTS TABLE")
    print("=" * 120)
    print(final_df.to_string(index=False))
    print(f"\n[INFO] Summary CSV: {summary_path}")
    print(f"[INFO] Final combined CSV: {final_csv}")
    print(f"[INFO] Final combined TXT: {final_txt}")
    print(f"[DONE] All outputs under: {args.outdir}")


if __name__ == "__main__":
    main()