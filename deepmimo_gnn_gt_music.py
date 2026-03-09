#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepMIMO + GNN + MUSIC + CRB + n_select/group_size sweep

What this script does:
- Load DeepMIMO channels once
- Select n_select subcarriers using:
    - correlation
    - fisher
    - doptimal
- Group the selected subcarriers into groups of size group_size
- Build one graph per sample:
    - each node = one subcarrier group
    - node feature = covariance feature averaged over that group
- Train GNN on GT AoA labels
- Evaluate GNN and MUSIC against GT AoA
- Compute approximate grouped CRB
- Save:
    - per-run metrics.csv
    - selection_groups.json
    - training curve
    - scatter plots
    - time comparison
    - summary.json
    - summary_results.csv
    - all_methods_results.csv / txt

Example:
python deepmimo_gnn_grouped_sweep.py \
  --scenario asu_campus_3p5 \
  --candidate-subc 64 \
  --subc-start 0 \
  --n-select-list 20,24,30,40 \
  --group-min 2 \
  --group-max 10 \
  --run-all-methods \
  --methods correlation fisher doptimal \
  --bs-x 8 --bs-y 1 \
  --ue-x 1 --ue-y 1 \
  --epochs 30 \
  --batch-size 64 \
  --conv-type sage \
  --outdir results_deepmimo_grouped

Install:
pip install numpy pandas matplotlib scipy
pip install torch torch-geometric
pip install --pre deepmimo
"""

import os
import json
import time
import argparse
from typing import Optional, Tuple, List, Dict

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


def mae_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


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
    """
    Returns train_idx, val_idx, test_idx
    """
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
# Steering / MUSIC / CRB
# ============================================================

def steering_vector_ula(num_ant: int, angle_rad: float, d_over_lambda: float = 0.5) -> np.ndarray:
    n = np.arange(num_ant)
    return np.exp(1j * 2 * np.pi * d_over_lambda * n * np.sin(angle_rad))[:, None]


def steering_derivative_ula(num_ant: int, angle_rad: float, d_over_lambda: float = 0.5) -> np.ndarray:
    n = np.arange(num_ant)
    a = np.exp(1j * 2 * np.pi * d_over_lambda * n * np.sin(angle_rad))
    coeff = 1j * 2 * np.pi * d_over_lambda * n * np.cos(angle_rad)
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
        angle_grid_deg = np.linspace(-180, 180, 1441)

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
    h_sample: np.ndarray,              # [M, S_selected]
    groups_sc: List[List[int]],        # relative indices inside selected subcarriers
    num_sources: int = 1,
    angle_grid_deg: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Group-aware MUSIC:
    - For each group, average covariance over the group's selected subcarriers
    - Compute MUSIC spectrum for each group
    - Aggregate spectra by mean
    """
    if angle_grid_deg is None:
        angle_grid_deg = np.linspace(-180, 180, 1441)

    group_spectra = []
    for g in groups_sc:
        hg = h_sample[:, g]  # [M, |group|]
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
    h_sample: np.ndarray,              # [M, S_selected]
    theta_deg: float,
    groups_sc: List[List[int]],
    d_over_lambda: float = 0.5,
    num_sources: int = 1
) -> float:
    """
    Approximate grouped CRB:
    - each group contributes a FIM term using SNR proxy of the group's covariance
    - sum contributions over groups
    """
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
):
    """
    Returns:
        channels_candidate: [N, ue_ant, bs_ant, candidate_subc] or [N, bs_ant, candidate_subc]
        labels_deg: [N]
        meta: dict
    """
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

    labels_deg_all = aoa_az[:, 0].astype(np.float32)
    valid_mask = np.isfinite(labels_deg_all)

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
        channels = channels[idxs]
        labels_deg = labels_deg_all[idxs]
    else:
        labels_deg = np.asarray(safe_getattr(dataset, "aoa_az"))[:, 0].astype(np.float32)

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
    }
    return channels, labels_deg, meta


# ============================================================
# Channel helpers
# ============================================================

def squeeze_channel_to_bs_sc(channels: np.ndarray) -> np.ndarray:
    """
    Input:
        [N, ue_ant, bs_ant, n_sc] or [N, bs_ant, n_sc]
    Output:
        [N, bs_ant, n_sc]
    """
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


def cov_avg_over_subcarriers_list(h_sample: np.ndarray, sc_list: List[int]) -> np.ndarray:
    """
    h_sample: [M, S]
    sc_list must contain LOCAL indices inside h_sample, i.e. 0...(S-1)
    """
    M, S = h_sample.shape
    R_sum = np.zeros((M, M), dtype=np.complex128)

    for sc in sc_list:
        if sc < 0 or sc >= S:
            raise IndexError(
                f"Subcarrier index {sc} is out of bounds for h_sample with {S} selected subcarriers. "
                f"Make sure group indices are local to the selected tensor."
            )
        x = h_sample[:, [sc]]
        R_sum += (x @ np.conjugate(x.T))

    return R_sum / max(1, len(sc_list))

# ============================================================
# Subcarrier selection
# ============================================================

def feature_per_subcarrier(ch_bs_sc: np.ndarray) -> np.ndarray:
    """
    ch_bs_sc: [N, M, S]
    returns scalar feature matrix [N, S]
    """
    return np.mean(np.abs(ch_bs_sc), axis=1)


def compute_subcarrier_signatures(ch_bs_sc: np.ndarray, trace_norm: bool = True) -> np.ndarray:
    """
    Signature per subcarrier from all training samples:
    ch_bs_sc: [N, M, S]
    returns [S, F]
    """
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
    """
    Fisher-style discriminative score using GT AoA bins.
    """
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


def dopt_greedy_select_from_signatures(sigs: np.ndarray, n_select: int, proj_dim: int = 32,
                                       eps: float = 1e-6, seed: int = 7):
    """
    D-optimal greedy selection on signature vectors.
    sigs: [S, F]
    """
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
    """
    Group selected subcarriers into groups of size group_size using greedy similarity linkage.
    """
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
    ch_train_bs_sc: np.ndarray,      # [Ntrain, M, S]
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
    """
    Selection on training set only.

    Returns:
        selected_rel_idx_sorted: selected indices inside candidate window, sorted
        groups_sc_local: grouped LOCAL indices relative to the selected tensor
        groups_sc_candidate: grouped indices in candidate-window space
        scores: ranking scores or None
    """
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

    # keep original candidate-space grouping first
    groups_sc_candidate = greedy_group_by_similarity(selected_rel_idx, sim, group_size=group_size)

    # sort selected indices because channel slicing uses this exact order
    selected_rel_idx_sorted = np.array(sorted(selected_rel_idx), dtype=int)

    # map candidate-space group indices -> local indices in sliced tensor
    local_map = {sc: i for i, sc in enumerate(selected_rel_idx_sorted.tolist())}
    groups_sc_local = [[local_map[sc] for sc in g] for g in groups_sc_candidate]

    return selected_rel_idx_sorted, groups_sc_local, groups_sc_candidate, scores
# ============================================================
# Graph conversion
# ============================================================

def channel_to_graph_grouped(
    ch_selected: np.ndarray,         # [M, S_selected]
    groups_sc: List[List[int]],
    label_deg: float,
    topk: int = 0,
    trace_norm: bool = True
) -> Data:
    """
    Node = one subcarrier group
    Feature = covariance feature averaged over that group
    """
    ch_selected = normalize_complex_channel(ch_selected)

    node_feats = []
    for g in groups_sc:
        hg = ch_selected[:, g]     # [M, group_size]

        feat = np.concatenate([
            np.real(hg).reshape(-1),
            np.imag(hg).reshape(-1)
        ])

        node_feats.append(feat.astype(np.float32))

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

    return Data(
        x=torch.tensor(x_nodes, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([label_deg / 90.0], dtype=torch.float32)
    )


def build_graph_dataset_grouped(
    ch_bs_sc: np.ndarray,            # [N, M, S_selected]
    labels_deg: np.ndarray,
    groups_sc: List[List[int]],
    topk: int = 0,
    trace_norm: bool = True
) -> List[Data]:
    dataset = []
    for i in range(len(labels_deg)):
        dataset.append(
            channel_to_graph_grouped(
                ch_selected=ch_bs_sc[i],
                groups_sc=groups_sc,
                label_deg=float(labels_deg[i]),
                topk=topk,
                trace_norm=trace_norm
            )
        )
    return dataset


# ============================================================
# GNN
# ============================================================

class GNNRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "sage",
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = dropout
        self.conv_type = conv_type.lower()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if self.conv_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif self.conv_type == "gatv2":
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False))
            else:
                raise ValueError("conv_type must be 'sage' or 'gatv2'")

        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        out = self.head(x).squeeze(-1)
        return out


# ============================================================
# Train / eval
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y.view(-1)
        loss = F.mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        n = target.numel()
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    model.eval()

    ys = []
    preds = []
    total_time = 0.0
    total_n = 0

    for batch in loader:
        batch = batch.to(device)

        t0 = time.perf_counter()
        pred = model(batch)
        t1 = time.perf_counter()

        ys.append((batch.y.view(-1) * 90.0).cpu().numpy())
        preds.append((pred * 90.0).cpu().numpy())

        total_time += (t1 - t0)
        total_n += batch.y.numel()

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)

    return ys, preds, {
        "mae": mae_deg(ys, preds),
        "rmse": rmse_deg(ys, preds),
        "infer_time_total_sec": total_time,
        "infer_time_per_sample_sec": total_time / max(total_n, 1),
    }


def run_music_dataset_grouped(
    ch_bs_sc: np.ndarray,            # [N, M, S_selected]
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
        "mae": mae_deg(labels_deg, preds),
        "rmse": rmse_deg(labels_deg, preds),
        "infer_time_total_sec": total_time,
        "infer_time_per_sample_sec": total_time / max(len(preds), 1),
        "avg_crb_deg": avg_crb_deg,
    }


# ============================================================
# Plotting
# ============================================================

def plot_training_curve(train_losses, val_maes, outpath):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_maes, label="Val MAE")
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


def plot_time_comparison(gnn_ms, music_ms, outpath):
    plt.figure(figsize=(6, 5))
    plt.bar(["GNN", "MUSIC"], [gnn_ms, music_ms])
    plt.ylabel("Time per sample (ms)")
    plt.title("Inference Time Comparison")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_overall_doa_comparison(experiment_root: str, results_index: list):
    """
    One figure per method with rows for all valid (n_select, group_size) runs.
    """
    by_method = {}
    for r in results_index:
        by_method.setdefault(r["method"], []).append(r)

    for method, runs in by_method.items():
        runs = sorted(runs, key=lambda x: (x["n_select"], x["group_size"]))
        series = []

        for r in runs:
            metrics_path = os.path.join(r["run_dir"], "metrics.csv")
            if not os.path.exists(metrics_path):
                continue
            df = pd.read_csv(metrics_path).sort_values("sample").reset_index(drop=True)
            series.append({
                "n_select": r["n_select"],
                "group_size": r["group_size"],
                "sample": df["sample"].values,
                "gt": df["gt_doa"].values,
                "music": df["music_doa"].values,
                "gnn": df["gnn_doa"].values,
                "gnn_err": np.abs(df["gnn_doa"].values - df["gt_doa"].values),
                "music_err": np.abs(df["music_doa"].values - df["gt_doa"].values),
            })

        if len(series) == 0:
            continue

        nrows = len(series)
        plt.figure(figsize=(15, 3.5 * nrows))
        for i, s in enumerate(series):
            ax1 = plt.subplot(nrows, 2, 2 * i + 1)
            ax1.plot(s["sample"], s["gt"], label="GT")
            ax1.plot(s["sample"], s["music"], label="MUSIC")
            ax1.plot(s["sample"], s["gnn"], label="GNN")
            ax1.set_title(f"{method} | n_select={s['n_select']} | g={s['group_size']} (DoA)")
            ax1.set_xlabel("Test sample")
            ax1.set_ylabel("AoA (deg)")
            ax1.grid(True)
            ax1.legend()

            ax2 = plt.subplot(nrows, 2, 2 * i + 2)
            ax2.plot(s["sample"], s["music_err"], label="|MUSIC-GT|")
            ax2.plot(s["sample"], s["gnn_err"], label="|GNN-GT|")
            ax2.set_title(f"{method} | n_select={s['n_select']} | g={s['group_size']} (Abs Error)")
            ax2.set_xlabel("Test sample")
            ax2.set_ylabel("Error (deg)")
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()
        out_path = os.path.join(experiment_root, f"overall_doa_comparison_{method}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved overall DoA comparison: {out_path}")


# ============================================================
# Single run
# ============================================================

def run_one_config(
    ch_bs_sc: np.ndarray,            # [N, M, S]
    labels_deg: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    experiment_root: str,
    selection_method: str,
    n_select: int,
    group_size: int,
    angle_grid_deg: np.ndarray
):
    """
    One run = one method + one n_select + one group_size
    """
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

    # ------------------------------------------------
    # Selection on training set only
    # ------------------------------------------------
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

    # ------------------------------------------------
    # Slice selected subcarriers
    # ------------------------------------------------
    ch_selected = ch_bs_sc[:, :, selected_rel_idx]   # [N, M, n_select]

    # ------------------------------------------------
    # Build grouped graph dataset
    # ------------------------------------------------
    all_graphs = build_graph_dataset_grouped(
        ch_bs_sc=ch_selected,
        labels_deg=labels_deg,
        groups_sc=groups_sc,
        topk=args.topk,
        trace_norm=args.trace_norm
    )

    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs = [all_graphs[i] for i in val_idx]
    test_graphs = [all_graphs[i] for i in test_idx]

    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    test_ch_selected = ch_selected[test_idx]
    test_labels = labels_deg[test_idx]

    # ------------------------------------------------
    # Model
    # ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = train_graphs[0].x.shape[1]

    model = GNNRegressor(
        in_dim=in_dim,
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

    best_val_mae = float("inf")
    best_state = None
    train_losses = []
    val_maes = []

    # ------------------------------------------------
    # Train
    # ------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        _, _, val_metrics = evaluate_gnn(model, val_loader, device)

        train_losses.append(train_loss)
        val_maes.append(val_metrics["mae"])

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        train_rmse_deg = (train_loss ** 0.5) * 90.0
        print(
            f"[{selection_method} | n_select={n_select} | g={group_size} | Epoch {epoch:03d}] "
            f"Train MSE(norm)={train_loss:.6f} | Train RMSE(deg)={train_rmse_deg:.4f} | "
            f"Val MAE={val_metrics['mae']:.4f} | Val RMSE={val_metrics['rmse']:.4f}"
        )

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
            "selection_method": selection_method,
            "n_select": n_select,
            "group_size": group_size,
            "selected_relative_subcarriers": selected_rel_idx.tolist(),
            "selected_absolute_subcarriers": selected_abs_idx.tolist(),
            "groups_sc_relative_to_selected": groups_sc,
        },
        os.path.join(gnn_data_dir, "gnn_model_state.pt")
    )

    # ------------------------------------------------
    # Evaluate GNN
    # ------------------------------------------------
    y_true_gnn, y_pred_gnn, gnn_metrics = evaluate_gnn(model, test_loader, device)
    np.save(os.path.join(gnn_data_dir, "gnn_predictions.npy"), y_pred_gnn)

    # ------------------------------------------------
    # Evaluate MUSIC + grouped CRB
    # ------------------------------------------------
    y_pred_music, spectra_all, crb_deg_all, music_metrics = run_music_dataset_grouped(
        ch_bs_sc=test_ch_selected,
        labels_deg=test_labels,
        groups_sc=groups_sc,
        angle_grid_deg=angle_grid_deg
    )

    # ------------------------------------------------
    # Save per-sample metrics and plots
    # ------------------------------------------------
    rows = []
    for i in range(len(test_labels)):
        rows.append({
            "sample": int(i),
            "gt_doa": float(test_labels[i]),
            "gnn_doa": float(y_pred_gnn[i]),
            "music_doa": float(y_pred_music[i]),
            "gnn_abs_error": float(abs(y_pred_gnn[i] - test_labels[i])),
            "music_abs_error": float(abs(y_pred_music[i] - test_labels[i])),
            "gnn_sq_error": float((y_pred_gnn[i] - test_labels[i]) ** 2),
            "music_sq_error": float((y_pred_music[i] - test_labels[i]) ** 2),
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

    # sample-wise spectrum plots
    for i in range(min(len(test_labels), args.max_plot_samples)):
        Pagg = spectra_all[i]
        gt = test_labels[i]
        gnn = y_pred_gnn[i]
        music = y_pred_music[i]

        plt.figure(figsize=(7, 5))
        plt.semilogy(angle_grid_deg, Pagg / (np.max(Pagg) + 1e-12), label="Grouped MUSIC")
        plt.axvline(gt, linestyle="-", label="GT")
        plt.axvline(music, linestyle="--", label="MUSIC")
        plt.axvline(gnn, linestyle=":", label="GNN")
        plt.xlabel("Angle (degree)")
        plt.ylabel("Spectrum (normalized)")
        plt.title(f"Test sample {i} | {selection_method} | sel={n_select} | g={group_size}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(gnn_figure_dir, f"sample_{i}_gnn_music_comparison.png"), dpi=200)
        plt.close()

    # ------------------------------------------------
    # Save artifacts
    # ------------------------------------------------
    plot_training_curve(
        train_losses,
        val_maes,
        os.path.join(run_dir, "training_curve.png")
    )

    plot_scatter(
        test_labels,
        y_pred_gnn,
        f"GNN: GT vs Predicted AoA ({selection_method}, sel={n_select}, g={group_size})",
        os.path.join(run_dir, "scatter_gnn.png")
    )

    plot_scatter(
        test_labels,
        y_pred_music,
        f"MUSIC: GT vs Predicted AoA ({selection_method}, sel={n_select}, g={group_size})",
        os.path.join(run_dir, "scatter_music.png")
    )

    plot_time_comparison(
        gnn_metrics["infer_time_per_sample_sec"] * 1000.0,
        music_metrics["infer_time_per_sample_sec"] * 1000.0,
        os.path.join(run_dir, "time_comparison.png")
    )

    # DoA curves
    # DoA curves (plot only a subset for readability)
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
        "gnn_mae": gnn_metrics["mae"],
        "gnn_rmse": gnn_metrics["rmse"],
        "gnn_time_per_sample_ms": gnn_metrics["infer_time_per_sample_sec"] * 1000.0,
        "music_mae": music_metrics["mae"],
        "music_rmse": music_metrics["rmse"],
        "music_time_per_sample_ms": music_metrics["infer_time_per_sample_sec"] * 1000.0,
        "avg_crb_deg": music_metrics["avg_crb_deg"],
    }
    save_json(run_summary, os.path.join(run_dir, "summary.json"))

    print(
        f"[DONE] {selection_method} | n_select={n_select} | g={group_size} | "
        f"GNN MAE={gnn_metrics['mae']:.4f} | MUSIC MAE={music_metrics['mae']:.4f} | "
        f"GNN t={gnn_metrics['infer_time_per_sample_sec']*1000:.4f} ms | "
        f"MUSIC t={music_metrics['infer_time_per_sample_sec']*1000:.4f} ms | "
        f"CRB={music_metrics['avg_crb_deg']:.4f}"
    )

    return run_dir, {
        "selection_method": selection_method,
        "n_select": n_select,
        "group_size": group_size,
        "num_groups": len(groups_sc),
        "gnn_mae": gnn_metrics["mae"],
        "gnn_rmse": gnn_metrics["rmse"],
        "gnn_time_per_sample": gnn_metrics["infer_time_per_sample_sec"],
        "music_mae": music_metrics["mae"],
        "music_rmse": music_metrics["rmse"],
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

    # scenario / antennas
    parser.add_argument("--scenario", type=str, default="asu_campus_3p5")
    parser.add_argument("--bs-x", type=int, default=8)
    parser.add_argument("--bs-y", type=int, default=1)
    parser.add_argument("--ue-x", type=int, default=1)
    parser.add_argument("--ue-y", type=int, default=1)

    # candidate pool
    parser.add_argument("--candidate-subc", type=int, default=64)
    parser.add_argument("--subc-start", type=int, default=0)

    # sweep
    parser.add_argument("--n-select-list", type=str, default="20")
    parser.add_argument("--group-min", type=int, default=2)
    parser.add_argument("--group-max", type=int, default=10)

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["correlation", "fisher", "doptimal"],
        choices=["contiguous", "correlation", "fisher", "doptimal"]
    )
    parser.add_argument("--run-all-methods", action="store_true")

    # filtering
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--los-mode", type=str, default="all", choices=["all", "los", "nlos"])

    # split
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # selection parameters
    parser.add_argument("--corr-top-p", type=int, default=10)
    parser.add_argument("--fisher-bins", type=int, default=18)
    parser.add_argument("--dopt-proj-dim", type=int, default=32)
    parser.add_argument("--dopt-eps", type=float, default=1e-6)
    parser.add_argument("--dopt-seed", type=int, default=7)

    # graph/model
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--trace-norm", action="store_true")
    parser.add_argument("--conv-type", type=str, default="sage", choices=["sage", "gatv2"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # train
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    # plotting
    parser.add_argument("--max-plot-samples", type=int, default=30)
    parser.add_argument("--max-doa-curve-samples", type=int, default=300)

    # output
    parser.add_argument("--outdir", type=str, default="results_deepmimo_grouped_sweep")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    if args.group_min > args.group_max:
        raise ValueError("group_min cannot be larger than group_max")

    n_select_list = parse_int_list(args.n_select_list)
    if len(n_select_list) == 0:
        raise ValueError("n-select-list is empty. Example: --n-select-list 20,24,30,40")

    methods = args.methods if args.run_all_methods else [args.methods[0]]
    group_values = list(range(args.group_min, args.group_max + 1))

    # --------------------------------------------------------
    # Load candidate channels once
    # --------------------------------------------------------
    channels_candidate, labels_deg, meta = load_deepmimo_candidate_channels(
        scenario=args.scenario,
        bs_shape=(args.bs_x, args.bs_y),
        ue_shape=(args.ue_x, args.ue_y),
        candidate_subc=args.candidate_subc,
        subc_start=args.subc_start,
        max_users=args.max_users,
        los_mode=args.los_mode,
    )

    print("[INFO] Candidate data loaded")
    print(json.dumps(meta, indent=2))

    ch_bs_sc = squeeze_channel_to_bs_sc(channels_candidate)  # [N, M, S]
    N, M, S = ch_bs_sc.shape

    train_idx, val_idx, test_idx = manual_split_indices(
        n=N,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed
    )

    print(f"[INFO] Dataset split: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

    summary_rows = []
    results_index = []
    final_table_rows = []
    skipped_rows = []
    angle_grid_deg = np.linspace(-180, 180, 1441)

    # --------------------------------------------------------
    # Sweep
    # --------------------------------------------------------
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
                    angle_grid_deg=angle_grid_deg
                )

                summary_rows.append(summ)
                results_index.append({
                    "method": method,
                    "n_select": nsel,
                    "group_size": gsize,
                    "run_dir": run_dir
                })

                final_table_rows.append({
                    "Method": pretty_method_name(method),
                    "nselect": nsel,
                    "GroupSize": gsize,
                    "NumGroups": summ["num_groups"],
                    "GNN (ms)": summ["gnn_time_per_sample"] * 1000.0,
                    "MUSIC (ms)": summ["music_time_per_sample"] * 1000.0,
                    "GNN MAE": summ["gnn_mae"],
                    "GNN RMSE": summ["gnn_rmse"],
                    "MUSIC MAE": summ["music_mae"],
                    "MUSIC RMSE": summ["music_rmse"],
                    "CRB": summ["avg_crb_deg"],
                    "Selected Subcarriers": ",".join(map(str, summ["selected_subcarriers"]))
                })

    if len(summary_rows) == 0:
        raise RuntimeError("No valid runs were executed. Check n-select-list and group range.")

    # --------------------------------------------------------
    # Save summaries
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Global comparison plots
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for method in methods:
        sdf = summary_df[summary_df["selection_method"] == method]
        if len(sdf) == 0:
            continue
        for gsize in sorted(sdf["group_size"].unique()):
            sdfg = sdf[sdf["group_size"] == gsize].sort_values("n_select")
            plt.plot(sdfg["n_select"], sdfg["gnn_mae"], label=f"{method} g={gsize} GNN MAE")
            plt.plot(sdfg["n_select"], sdfg["music_mae"], label=f"{method} g={gsize} MUSIC MAE")
    plt.xlabel("n_select")
    plt.ylabel("MAE (deg)")
    plt.title("GNN / MUSIC MAE vs n_select")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "comparison_mae.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for method in methods:
        sdf = summary_df[summary_df["selection_method"] == method]
        if len(sdf) == 0:
            continue
        for gsize in sorted(sdf["group_size"].unique()):
            sdfg = sdf[sdf["group_size"] == gsize].sort_values("n_select")
            plt.plot(sdfg["n_select"], sdfg["gnn_time_per_sample"], label=f"{method} g={gsize} GNN")
            plt.plot(sdfg["n_select"], sdfg["music_time_per_sample"], label=f"{method} g={gsize} MUSIC")
    plt.xlabel("n_select")
    plt.ylabel("Time per sample (sec)")
    plt.title("Inference time vs n_select")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "comparison_time.png"), dpi=200)
    plt.close()

    plot_overall_doa_comparison(args.outdir, results_index)

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