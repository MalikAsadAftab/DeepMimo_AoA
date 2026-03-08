import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepmimo as dm


# -----------------------------
# Output folder
# -----------------------------
out_dir = "MUSIC_EXP"
os.makedirs(out_dir, exist_ok=True)


# -----------------------------
# MUSIC helper functions
# -----------------------------
def find_specified_peaks(phi, PS, L):
    phi_in = phi[(phi >= -np.pi / 2) & (phi <= np.pi / 2)]
    PS_in = PS[(phi >= -np.pi / 2) & (phi <= np.pi / 2)]

    # fallback: global max if no peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.abs(PS_in))
    if len(peaks) == 0:
        return np.array([phi_in[np.argmax(np.abs(PS_in))]])

    sorted_peaks = peaks[np.argsort(np.abs(PS_in)[peaks])]
    return phi_in[sorted_peaks[-L:]]


def music(a, R, phi, L):
    PS = np.zeros(a.shape[1], dtype=complex)
    D, Q = np.linalg.eigh(R)
    Qn = Q[:, 0:R.shape[0] - L]
    Rn = Qn @ np.conjugate(Qn.T)

    for t in range(len(phi)):
        a_tmp = np.reshape(a[:, t], (-1, 1))
        PS[t] = 1.0 / (np.conjugate(a_tmp.T) @ Rn @ a_tmp).item()

    peaks = find_specified_peaks(phi, PS, L)
    return PS, peaks


def estimate_music_aoa_from_user_channel(H_user, lam, d, L=1):
    """
    H_user shape expected: (16, 64)
    Build covariance averaged over subcarriers, then run 1D MUSIC.
    """
    M, K = H_user.shape

    # covariance averaged over subcarriers
    R = np.zeros((M, M), dtype=np.complex128)
    for k in range(K):
        h = H_user[:, k].reshape(-1, 1)
        R += h @ np.conjugate(h.T)
    R /= K

    # trace normalization
    tr = np.trace(R)
    if np.abs(tr) > 1e-12:
        R = R / tr

    phi = np.arange(-np.pi, np.pi, np.pi / 360)
    m = np.arange(M).reshape(-1, 1)

    # ULA steering matrix
    a = np.exp(-1j * 2 * np.pi / lam * d * np.outer(m, np.sin(phi)))

    PS, peaks = music(a, R, phi, L)
    doa_deg = np.rad2deg(peaks[0])
    doa_deg = float(np.clip(doa_deg, -90.0, 90.0))
    return doa_deg, phi, PS


# -----------------------------
# Load DeepMIMO and generate channels
# -----------------------------
scenario = "asu_campus_3p5"
dataset = dm.load(scenario)

ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [16, 1]
ch_params.bs_antenna.spacing = 0.5
ch_params.ue_antenna.shape = [1, 1]
ch_params.freq_domain = 1
ch_params.ofdm.subcarriers = 64
ch_params.ofdm.selected_subcarriers = list(range(64))
ch_params.ofdm.bandwidth = 10e6

dataset.compute_channels(ch_params)

H = dataset.channels if hasattr(dataset, "channels") else dataset.channel

print("Channel shape:", H.shape)

# carrier frequency of asu_campus_3p5
f = 3.5e9
c = 3e8
lam = c / f
d = lam / 2

# valid users only
valid_mask = (dataset.los != -1) & np.isfinite(dataset.power[:, 0]) & np.isfinite(dataset.aoa_az[:, 0])
valid_idx = np.where(valid_mask)[0]

print("Valid users:", len(valid_idx))

# use a subset first for speed
max_users = 300
eval_idx = valid_idx[:max_users]

rows = []

for n, i in enumerate(eval_idx):
    H_user = H[i]  # expected shape like (1, 16, 64)
    H_user = np.squeeze(H_user, axis=0)  # -> (16, 64)

    finite_paths = np.where(np.isfinite(dataset.power[i]) & np.isfinite(dataset.aoa_az[i]))[0]
    if len(finite_paths) == 0:
        continue

    best_idx = finite_paths[np.argmax(dataset.power[i, finite_paths])]
    gt_aoa = float(dataset.aoa_az[i, best_idx])

    # skip GT angles outside MUSIC search region
    if gt_aoa < -90 or gt_aoa > 90:
        continue

    music_aoa, phi, PS = estimate_music_aoa_from_user_channel(H_user, lam=lam, d=d, L=1)

    rows.append({
        "user_index": int(i),
        "los": int(dataset.los[i]),
        "gt_aoa": gt_aoa,
        "music_aoa": music_aoa,
        "abs_error": abs(music_aoa - gt_aoa),
        "sq_error": (music_aoa - gt_aoa) ** 2,
    })

    if n < 5:
        print(f"user={i} | GT={gt_aoa:.3f} | MUSIC={music_aoa:.3f} | LOS={dataset.los[i]}")

df = pd.DataFrame(rows)

csv_path = os.path.join(out_dir, "deepmimo_music_baseline_metrics.csv")
df.to_csv(csv_path, index=False)

mae = df["abs_error"].mean()
rmse = np.sqrt(df["sq_error"].mean())

print("\n=== MUSIC BASELINE RESULTS ===")
print("Samples used:", len(df))
print(f"MAE  = {mae:.4f} deg")
print(f"RMSE = {rmse:.4f} deg")

# plot GT vs MUSIC
plt.figure(figsize=(12, 5))
plt.plot(df["gt_aoa"].values, label="Ground Truth AoA")
plt.plot(df["music_aoa"].values, label="MUSIC AoA")
plt.xlabel("Sample Index")
plt.ylabel("AoA (deg)")
plt.title("DeepMIMO: Ground Truth vs MUSIC")
plt.legend()
plt.grid(True)
plt.tight_layout()
gt_vs_music_path = os.path.join(out_dir, "deepmimo_gt_vs_music.png")
plt.savefig(gt_vs_music_path)
plt.close()

# abs error plot
plt.figure(figsize=(12, 4))
plt.plot(df["abs_error"].values, label="|MUSIC - GT|")
plt.xlabel("Sample Index")
plt.ylabel("Absolute Error (deg)")
plt.title("DeepMIMO MUSIC Absolute Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
abs_error_path = os.path.join(out_dir, "deepmimo_music_abs_error.png")
plt.savefig(abs_error_path)
plt.close()

print("Saved:")
print(f" - {csv_path}")
print(f" - {gt_vs_music_path}")
print(f" - {abs_error_path}")