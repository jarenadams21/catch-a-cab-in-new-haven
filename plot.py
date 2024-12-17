"""
DISCLAIMER:
This visualization script is designed to show data from a toy model code that 
is NOT a full simulation of QCD, FLRW cosmology, or gravitational waves. 
It uses synthetic fields and simplistic PDE steps. Actual research would require:

- Real lattice QCD EoS data (e.g., from the HotQCD collaboration).
- Sophisticated PDE and Boltzmann solvers for momentum-dependent distributions.
- Proper cosmological modeling of gravitational waves and metric perturbations.

This script simply demonstrates how one might load and visualize 4D snapshots 
over time with multiple fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os
import pandas as pd

# Grid and field configuration must match the simulation code
NX = 100
NY = 100
num_fields = 7  # photon, neutrino, energy, a_field, h_field, metric_field, gw_field

# Load global evolution data
if not os.path.exists("results.csv"):
    raise FileNotFoundError("results.csv not found. Run the simulation first.")

df = pd.read_csv("results.csv")
time = df["time"]
a_scale = df["a"]

# Load snapshots
files = sorted(glob.glob("snapshots/fields_*.bin"))
if len(files) == 0:
    raise FileNotFoundError("No snapshot files found in snapshots/. Run the simulation first.")

data_list = []
for fname in files:
    raw = np.fromfile(fname, dtype='<f8')
    # shape: (num_fields, NY, NX)
    arr = raw.reshape(num_fields, NY, NX)
    data_list.append(arr)

data = np.array(data_list) # shape: (num_times, num_fields, NY, NX)
times = np.arange(len(data))

field_names = [
    "Photon Density", 
    "Neutrino Density", 
    "Energy Density", 
    "ALP Field (a)", 
    "Higgs-like Field (h)", 
    "Metric Perturbation",
    "GW Amplitude"
]

cmaps = ['inferno', 'plasma', 'magma', 'viridis', 'cividis', 'Greys', 'PuOr']

fig, axes = plt.subplots(2,4, figsize=(18,8))
axes = axes.flatten()

# Plot initial state for each field
ims = []
for i in range(num_fields):
    im = axes[i].imshow(data[0, i, :, :], origin='lower', cmap=cmaps[i])
    axes[i].set_title(field_names[i])
    fig.colorbar(im, ax=axes[i])
    ims.append(im)

# Last subplot for scale factor
ax_a = axes[-1]
ax_a.plot(time, a_scale, label="Scale Factor a(t)")
ax_a.set_title("Scale Factor Evolution")
ax_a.set_xlabel("Time (dimensionless)")
ax_a.set_ylabel("a(t)")
ax_a.grid(True)
ax_a.legend()

plt.tight_layout(rect=[0,0,1,0.95])
fig.suptitle("4D Field Visualization (Toy Model)")

# Slider for time steps
ax_time = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax=ax_time, label='Time Step', valmin=0, valmax=len(data)-1, valinit=0, valstep=1)

def update(val):
    t = int(time_slider.val)
    for i, im in enumerate(ims):
        im.set_data(data[t, i, :, :])
    fig.canvas.draw_idle()

time_slider.on_changed(update)

# Reminder that this is a demonstration only
print("NOTE: This visualization is from a toy model and not a realistic simulation.")

plt.show()
