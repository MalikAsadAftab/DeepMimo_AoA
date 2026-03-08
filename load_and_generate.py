import numpy as np
import deepmimo as dm

scenario = "asu_campus_3p5"
dataset = dm.load(scenario)

ch_params = dm.ChannelParameters()
ch_params.freq_domain = True
ch_params.bs_antenna.shape = [16, 1]
ch_params.ue_antenna.shape = [1, 1]
ch_params.ofdm.subcarriers = 64
ch_params.ofdm.bandwidth = 10e6

dataset.compute_channels(ch_params)

print("channels shape:", dataset.channels.shape if hasattr(dataset, "channels") else dataset.channel.shape)
print("power shape   :", dataset.power.shape)
print("aoa_az shape  :", dataset.aoa_az.shape)
print("los shape     :", dataset.los.shape)

# valid user = has at least one path and finite AoA/power
valid_mask = (dataset.los != -1) & np.isfinite(dataset.power[:, 0]) & np.isfinite(dataset.aoa_az[:, 0])
valid_idx = np.where(valid_mask)[0]

print("Total users       :", len(dataset.los))
print("Valid users       :", len(valid_idx))
print("Invalid users     :", len(dataset.los) - len(valid_idx))
print("First 10 valid idx:", valid_idx[:10])

# inspect first valid sample
i = valid_idx[0]
print("\nFirst valid user index:", i)
print("LOS status:", dataset.los[i])
print("Path powers:", dataset.power[i])
print("AoA azimuths:", dataset.aoa_az[i])
print("AoA elevations:", dataset.aoa_el[i])

# strongest valid path only
finite_paths = np.where(np.isfinite(dataset.power[i]) & np.isfinite(dataset.aoa_az[i]))[0]
best_local = finite_paths[np.argmax(dataset.power[i, finite_paths])]

gt_aoa_deg = dataset.aoa_az[i, best_local]
print("Strongest valid path index:", best_local)
print("GT AoA azimuth (deg):", gt_aoa_deg)