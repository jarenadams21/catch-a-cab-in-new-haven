import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob, os
import pandas as pd

# Must match the Rust code constants
NX = 64
NY = 64
NZ = 8
NW = 4
NUM_FIELDS = 9
field_names = [
    "Photon Density",
    "Axion Density",
    "Neutrino Density",
    "Energy Density",
    "Metric Perturbation",
    "Torsion XX",
    "Torsion XY",
    "Torsion XZ",
    "Chiral-Odd Component"
]

cmaps = ['inferno', 'plasma', 'magma', 'viridis', 'cividis', 'Greys', 'RdBu', 'PiYG', 'coolwarm']

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
    # shape: (NUM_FIELDS, NW, NZ, NY, NX)
    arr = raw.reshape(NUM_FIELDS, NW, NZ, NY, NX)
    data_list.append(arr)

data = np.array(data_list) # shape: (num_times, NUM_FIELDS, NW, NZ, NY, NX)
num_times = data.shape[0]

# Initial indices
t_idx = 0
w_idx = 0
z_idx = 0

fig, axes = plt.subplots(2,5, figsize=(20,8))
axes = axes.flatten()

# We'll visualize a subset of fields for clarity
# Let's pick Photon, Axion, Neutrino, Energy, Metric for main visualization
fields_to_show = [0,1,2,3,4]  # first five fields
ims = []
for i, f_id in enumerate(fields_to_show):
    # Slice initial W,Z
    im = axes[i].imshow(data[t_idx, f_id, w_idx, z_idx, :, :], origin='lower', cmap=cmaps[f_id])
    axes[i].set_title(field_names[f_id])
    fig.colorbar(im, ax=axes[i])
    ims.append(im)

# Last subplot for scale factor
ax_a = axes[-1]
ax_a.plot(time, a_scale, label="Scale Factor a(t)")
ax_a.set_title("Scale Factor")
ax_a.set_xlabel("Time")
ax_a.set_ylabel("a(t)")
ax_a.grid(True)
ax_a.legend()

plt.tight_layout(rect=[0,0,1,0.95])
fig.suptitle("5D Field Visualization (Tarski-Boltzmann, EC Torsion, SM Extension)")

# Sliders for time, W slice, Z slice
ax_time = plt.axes([0.2, 0.01, 0.6, 0.02], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax=ax_time, label='Time', valmin=0, valmax=num_times-1, valinit=0, valstep=1)

ax_w = plt.axes([0.2, 0.05, 0.27, 0.02], facecolor='lightgoldenrodyellow')
w_slider = Slider(ax=ax_w, label='W Dim', valmin=0, valmax=NW-1, valinit=0, valstep=1)

ax_z = plt.axes([0.6, 0.05, 0.27, 0.02], facecolor='lightgoldenrodyellow')
z_slider = Slider(ax=ax_z, label='Z Dim', valmin=0, valmax=NZ-1, valinit=0, valstep=1)

def update(val):
    t = int(time_slider.val)
    w = int(w_slider.val)
    z = int(z_slider.val)
    for i, f_id in enumerate(fields_to_show):
        ims[i].set_data(data[t, f_id, w, z, :, :])
    fig.canvas.draw_idle()

time_slider.on_changed(update)
w_slider.on_changed(update)
z_slider.on_changed(update)

print("Visualizing complex 5D dataset with Tarski-Boltzmann measure, EC torsion, and SM extension fields.")
plt.show()
