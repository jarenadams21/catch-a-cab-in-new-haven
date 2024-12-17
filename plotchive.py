import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import os

NX = 100
NY = 100
num_fields = 6  # photon, neutrino, energy, a_field, h_field, metric_field

# Load all snapshots:
files = sorted(glob.glob("snapshots/fields_*.bin"))
if len(files) == 0:
    raise FileNotFoundError("No snapshot files found in snapshots/")

data_list = []
for fname in files:
    raw = np.fromfile(fname, dtype='<f8') # little-endian float64
    # Each file: num_fields * NX * NY
    # shape: (num_fields, NY, NX)
    arr = raw.reshape(num_fields, NY, NX)
    data_list.append(arr)

data = np.array(data_list) # shape (num_times, num_fields, NY, NX)

times = np.arange(len(data))  # Not actual time in units, but snapshot index

# Field names for titles:
field_names = [
    "Photon Density", 
    "Neutrino Density", 
    "Energy Density", 
    "ALP Field (a)", 
    "Higgs-like Field (h)", 
    "Metric Perturbation"
]

# Create figure with subplots for each field
fig, axes = plt.subplots(2,3, figsize=(15,8))
axes = axes.flatten()

# Initial plot at time_idx=0
time_idx = 0
ims = []
cmaps = ['inferno', 'plasma', 'magma', 'viridis', 'cividis', 'RdPu']
for i, ax in enumerate(axes):
    im = ax.imshow(data[time_idx, i, :, :], origin='lower', cmap=cmaps[i])
    ax.set_title(field_names[i])
    fig.colorbar(im, ax=ax)
    ims.append(im)

plt.tight_layout(rect=[0,0,1,0.95])
fig.suptitle("4D Field Visualization (Time, Fields, X, Y)")

# Add a slider to move through time:
ax_time = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax=ax_time, label='Time Step', valmin=0, valmax=len(data)-1, valinit=0, valstep=1)

def update(val):
    t = int(time_slider.val)
    for i, im in enumerate(ims):
        im.set_data(data[t, i, :, :])
    fig.canvas.draw_idle()

time_slider.on_changed(update)

plt.show()
