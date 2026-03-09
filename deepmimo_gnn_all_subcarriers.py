#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepMIMO + GNN + MUSIC + CRB
All subcarriers, no grouping

What this script does:
- Load DeepMIMO channels
- Use all selected candidate subcarriers directly
- Build one graph per sample:
    - each node = one subcarrier
    - node feature = real/imag parts of the antenna response at that subcarrier
- Train GNN on GT AoA labels
- Evaluate GNN and MUSIC against GT AoA
- Compute approximate CRB
- Save:
    - metrics.csv
    - training_curve.png
    - scatter plots
    - time comparison
    - summary.json
    - summary_results.csv
    - all_methods_results.csv / txt

Example:
python deepmimo_gnn_all_subcarriers.py \
  --scenario asu_campus_3p5 \
  --candidate-subc 64 \
  --subc-start 0 \
  --bs-x 8 --bs-y 1 \
  --ue-x 1 --ue-y 1 \
  --epochs 30 \
  --batch-size 64 \
  --conv-type sage \
  --outdir results_deepmimo_all_subcarriers
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


def run_music_single_sample(
    h_sample: np.ndarray,   # [M, S]
    num_sources: int = 1,
    angle_grid_deg: Optional[np.ndarray] = None
):
    if angle_grid_deg is None:
        angle_grid_deg = np.linspace(-180, 180, 1441)

    R = estimate_covariance_from_subcarriers(h_sample)
    phi_deg, P = music_spectrum_ula(
        R=R,
        num_sources=num_sources,
        angle_grid_deg=angle_grid_deg
    )
    est = find_specified_peaks(phi_deg, P, L=1)[0]
    return float(est), P


def estimate_snr_proxy_from_cov(R: np.ndarray, L: int = 1, eps: float = 1e-12):
    vals = np.linalg.eigvalsh(R)
    vals = np.sort(np.real(vals))[::-1]
    M = len(vals)
    noise = float(np.mean(vals[L:])) if (M - L) > 0 else float(vals[-1])
    signal_pow = float(max(vals[0] - noise, 0.0))
    return signal_pow / (noise + eps)


def approx_crb_deg(
    h_sample: np.ndarray,   # [M, S]
    theta_deg: float,
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

    R = estimate_covariance_from_subcarriers(h_sample)
    snr = estimate_snr_proxy_from_cov(R, L=num_sources)

    num_snapshots = h_sample.shape[1]
    fim = 2.0 * max(num_snapshots, 1) * max(snr, 0.0) * deriv_term
    fim = max(fim, 1e-12)

    crb_rad2 = 1.0 / fim
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
    if channels.ndim == 4:
        return channels[:, 0, :, :]
    elif channels.ndim == 3:
        return channels
    else:
        raise ValueError(f"Unsupported channel shape: {channels.shape}")


def normalize_complex_channel(h: np.ndarray) -> np.ndarray:
    power = np.sqrt(np.mean(np.abs(h) ** 2) + 1e-12)
    return h / power


# ============================================================
# Graph conversion
# ============================================================

def channel_to_graph_all_subcarriers(
    ch_sample: np.ndarray,     # [M, S]
    label_deg: float,
    topk: int = 0
) -> Data:
    """
    Each node = one subcarrier
    Node feature = real/imag of antenna vector at that subcarrier
    """
    ch_sample = normalize_complex_channel(ch_sample)
    M, S = ch_sample.shape

    node_feats = []
    for s in range(S):
        hs = ch_sample[:, s]   # [M]
        feat = np.concatenate([np.real(hs), np.imag(hs)], axis=0)
        node_feats.append(feat.astype(np.float32))

    x_nodes = np.stack(node_feats, axis=0)   # [S, 2M]
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


def build_graph_dataset_all_subcarriers(
    ch_bs_sc: np.ndarray,     # [N, M, S]
    labels_deg: np.ndarray,
    topk: int = 0
) -> List[Data]:
    dataset = []
    for i in range(len(labels_deg)):
        dataset.append(
            channel_to_graph_all_subcarriers(
                ch_sample=ch_bs_sc[i],
                label_deg=float(labels_deg[i]),
                topk=topk
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

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pred = model(batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
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


def run_music_dataset(
    ch_bs_sc: np.ndarray,       # [N, M, S]
    labels_deg: np.ndarray,
    angle_grid_deg=None
):
    preds = []
    total_time = 0.0
    crb_deg_all = []
    spectra_all = []

    for i in range(ch_bs_sc.shape[0]):
        h = ch_bs_sc[i]

        t0 = time.perf_counter()
        pred, P = run_music_single_sample(
            h_sample=h,
            num_sources=1,
            angle_grid_deg=angle_grid_deg
        )
        t1 = time.perf_counter()

        preds.append(pred)
        total_time += (t1 - t0)
        spectra_all.append(P)

        try:
            crb_deg = approx_crb_deg(
                h_sample=h,
                theta_deg=float(labels_deg[i]),
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


# ============================================================
# Single run
# ============================================================

def run_all_subcarriers(
    ch_bs_sc: np.ndarray,        # [N, M, S]
    labels_deg: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    experiment_root: str,
    angle_grid_deg: np.ndarray
):
    N, M, S = ch_bs_sc.shape

    run_dir = os.path.join(experiment_root, "all_subcarriers_no_grouping")
    data_dir = os.path.join(run_dir, "data")
    gnn_figure_dir = os.path.join(run_dir, "gnn_figure")
    gnn_data_dir = os.path.join(run_dir, "gnn_data")
    ensure_dir(run_dir)
    ensure_dir(data_dir)
    ensure_dir(gnn_figure_dir)
    ensure_dir(gnn_data_dir)

    selected_rel_idx = np.arange(S, dtype=int)
    selected_abs_idx = selected_rel_idx + args.subc_start

    selection_meta = {
        "selection_method": "all_subcarriers",
        "n_select": int(S),
        "group_size": 1,
        "num_groups": int(S),
        "selected_relative_subcarriers": selected_rel_idx.tolist(),
        "selected_absolute_subcarriers": selected_abs_idx.tolist(),
    }
    save_json(selection_meta, os.path.join(gnn_data_dir, "selection_groups.json"))

    # ------------------------------------------------
    # Build graph dataset
    # ------------------------------------------------
    all_graphs = build_graph_dataset_all_subcarriers(
        ch_bs_sc=ch_bs_sc,
        labels_deg=labels_deg,
        topk=args.topk
    )

    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs = [all_graphs[i] for i in val_idx]
    test_graphs = [all_graphs[i] for i in test_idx]

    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    test_ch = ch_bs_sc[test_idx]
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
    train_start = time.perf_counter()

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
            f"[all_subcarriers | Epoch {epoch:03d}] "
            f"Train MSE(norm)={train_loss:.6f} | "
            f"Train RMSE(deg)={train_rmse_deg:.4f} | "
            f"Val MAE={val_metrics['mae']:.4f} | "
            f"Val RMSE={val_metrics['rmse']:.4f}"
        )

    train_end = time.perf_counter()
    total_training_time_sec = train_end - train_start

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
            "selected_relative_subcarriers": selected_rel_idx.tolist(),
            "selected_absolute_subcarriers": selected_abs_idx.tolist(),
        },
        os.path.join(gnn_data_dir, "gnn_model_state.pt")
    )

    # ------------------------------------------------
    # Evaluate GNN
    # ------------------------------------------------
    y_true_gnn, y_pred_gnn, gnn_metrics = evaluate_gnn(model, test_loader, device)
    np.save(os.path.join(gnn_data_dir, "gnn_predictions.npy"), y_pred_gnn)

    # ------------------------------------------------
    # Evaluate MUSIC + CRB
    # ------------------------------------------------
    y_pred_music, spectra_all, crb_deg_all, music_metrics = run_music_dataset(
        ch_bs_sc=test_ch,
        labels_deg=test_labels,
        angle_grid_deg=angle_grid_deg
    )

    # ------------------------------------------------
    # Save per-sample metrics
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
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # spectrum plots
    for i in range(min(len(test_labels), args.max_plot_samples)):
        P = spectra_all[i]
        gt = test_labels[i]
        gnn = y_pred_gnn[i]
        music = y_pred_music[i]

        plt.figure(figsize=(7, 5))
        plt.semilogy(angle_grid_deg, P / (np.max(P) + 1e-12), label="MUSIC")
        plt.axvline(gt, linestyle="-", label="GT")
        plt.axvline(music, linestyle="--", label="MUSIC")
        plt.axvline(gnn, linestyle=":", label="GNN")
        plt.xlabel("Angle (degree)")
        plt.ylabel("Spectrum (normalized)")
        plt.title(f"Test sample {i} | all subcarriers")
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
        "GNN: GT vs Predicted AoA (all subcarriers)",
        os.path.join(run_dir, "scatter_gnn.png")
    )

    plot_scatter(
        test_labels,
        y_pred_music,
        "MUSIC: GT vs Predicted AoA (all subcarriers)",
        os.path.join(run_dir, "scatter_music.png")
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
    plt.title(f"DoA comparison (all subcarriers) [first {num_plot} test samples]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "DoA_GT_vs_GNN_vs_MUSIC.png"), dpi=200)
    plt.close()

    run_summary = {
        "selection_method": "all_subcarriers",
        "n_select": int(S),
        "group_size": 1,
        "num_groups": int(S),

        "gnn_mae": gnn_metrics["mae"],
        "gnn_rmse": gnn_metrics["rmse"],
        "gnn_training_time_sec": total_training_time_sec,
        "gnn_training_time_min": total_training_time_sec / 60.0,
        "gnn_infer_time_total_sec": gnn_metrics["infer_time_total_sec"],
        "gnn_infer_time_per_sample_sec": gnn_metrics["infer_time_per_sample_sec"],
        "gnn_infer_time_per_sample_ms": gnn_metrics["infer_time_per_sample_sec"] * 1000.0,

        "music_mae": music_metrics["mae"],
        "music_rmse": music_metrics["rmse"],
        "music_calc_time_total_sec": music_metrics["infer_time_total_sec"],
        "music_calc_time_per_sample_sec": music_metrics["infer_time_per_sample_sec"],
        "music_calc_time_per_sample_ms": music_metrics["infer_time_per_sample_sec"] * 1000.0,

        "avg_crb_deg": music_metrics["avg_crb_deg"],
        "selected_absolute_subcarriers": selected_abs_idx.tolist(),
    }
    save_json(run_summary, os.path.join(run_dir, "summary.json"))

    print(
        f"[DONE] all_subcarriers | "
        f"GNN MAE={gnn_metrics['mae']:.4f} | "
        f"GNN RMSE={gnn_metrics['rmse']:.4f} | "
        f"MUSIC MAE={music_metrics['mae']:.4f} | "
        f"MUSIC RMSE={music_metrics['rmse']:.4f} | "
        f"Train time={total_training_time_sec:.4f} sec | "
        f"GNN infer/sample={gnn_metrics['infer_time_per_sample_sec']*1000:.4f} ms | "
        f"MUSIC calc/sample={music_metrics['infer_time_per_sample_sec']*1000:.4f} ms | "
        f"CRB={music_metrics['avg_crb_deg']:.4f}"
    )

    return run_dir, run_summary


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

    # candidate pool = full pool to use
    parser.add_argument("--candidate-subc", type=int, default=64)
    parser.add_argument("--subc-start", type=int, default=0)

    # filtering
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--los-mode", type=str, default="all", choices=["all", "los", "nlos"])

    # split
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # graph/model
    parser.add_argument("--topk", type=int, default=0)
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
    parser.add_argument("--outdir", type=str, default="results_deepmimo_all_subcarriers")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    # --------------------------------------------------------
    # Load candidate channels
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
    print(f"[INFO] Using all subcarriers with no grouping. Total subcarriers = {S}")

    angle_grid_deg = np.linspace(-180, 180, 1441)

    run_dir, summ = run_all_subcarriers(
        ch_bs_sc=ch_bs_sc,
        labels_deg=labels_deg,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        args=args,
        experiment_root=args.outdir,
        angle_grid_deg=angle_grid_deg
    )

    save_json(meta, os.path.join(args.outdir, "dataset_meta.json"))

    summary_df = pd.DataFrame([summ])
    summary_path = os.path.join(args.outdir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)

    final_df = pd.DataFrame([{
        "Method": "AllSubcarriers",
        "nselect": summ["n_select"],
        "GroupSize": summ["group_size"],
        "NumGroups": summ["num_groups"],

        "Training Time (sec)": summ["gnn_training_time_sec"],

        "GNN Inference Total (sec)": summ["gnn_infer_time_total_sec"],
        "GNN Inference / sample (ms)": summ["gnn_infer_time_per_sample_ms"],

        "MUSIC Calc Total (sec)": summ["music_calc_time_total_sec"],
        "MUSIC Calc / sample (ms)": summ["music_calc_time_per_sample_ms"],

        "GNN MAE": summ["gnn_mae"],
        "GNN RMSE": summ["gnn_rmse"],
        "MUSIC MAE": summ["music_mae"],
        "MUSIC RMSE": summ["music_rmse"],
        "CRB": summ["avg_crb_deg"],
        "Selected Subcarriers": ",".join(map(str, summ["selected_absolute_subcarriers"]))
    }])

    final_csv = os.path.join(args.outdir, "all_methods_results.csv")
    final_txt = os.path.join(args.outdir, "all_methods_results.txt")
    final_df.to_csv(final_csv, index=False)
    with open(final_txt, "w") as f:
        f.write(final_df.to_string(index=False))

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