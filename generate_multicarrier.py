import numpy as np
import deepmimo as dm

scenario = "asu_campus_3p5"
dataset = dm.load(scenario)

print("Loaded:", scenario)

ch_params = dm.ChannelParameters()
ch_params.bs_antenna.shape = [16, 1]
ch_params.bs_antenna.spacing = 0.5
ch_params.ue_antenna.shape = [1, 1]
ch_params.freq_domain = 1

# Important fix
ch_params.ofdm.subcarriers = 64
ch_params.ofdm.selected_subcarriers = list(range(64))
ch_params.ofdm.bandwidth = 10e6

print("\n=== CHANNEL PARAMS USED ===")
print(ch_params)

dataset.compute_channels(ch_params)

H = dataset.channels if hasattr(dataset, "channels") else dataset.channel

print("\n=== GENERATED CHANNELS ===")
print("channel shape:", H.shape)
print("power shape  :", dataset.power.shape)
print("aoa_az shape :", dataset.aoa_az.shape)
print("los shape    :", dataset.los.shape)

valid_mask = (dataset.los != -1) & np.isfinite(dataset.power[:, 0]) & np.isfinite(dataset.aoa_az[:, 0])
valid_idx = np.where(valid_mask)[0]

print("\nValid users:", len(valid_idx))
print("First valid users:", valid_idx[:10])

i = valid_idx[0]
print("\nInspect valid user:", i)
print("LOS:", dataset.los[i])
print("channel sample shape:", H[i].shape)
print("powers:", dataset.power[i])
print("aoa_az:", dataset.aoa_az[i])

finite_paths = np.where(np.isfinite(dataset.power[i]) & np.isfinite(dataset.aoa_az[i]))[0]
best_idx = finite_paths[np.argmax(dataset.power[i, finite_paths])]
print("Strongest-path GT AoA:", dataset.aoa_az[i, best_idx])