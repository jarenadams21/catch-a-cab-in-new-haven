import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os
import re

# Lattice sizes and parameters must match the Rust code
NX, NY, NZ, NW, NV = 4,4,4,2,2
NUM_FIELDS = 6  # (Φ, A0, Psi0..3)
D = 5

def load_fields(filename):
    size = NX*NY*NZ*NW*NV*NUM_FIELDS
    raw = np.fromfile(filename, dtype='<f8', count=size)
    arr = raw.reshape((NV,NW,NZ,NY,NX,NUM_FIELDS))
    return arr

def load_torsion(filename):
    t_size = NX*NY*NZ*NW*NV*D*D*D
    raw = np.fromfile(filename, dtype='<f8', count=t_size)
    arr = raw.reshape((NV,NW,NZ,NY,NX,D,D,D))
    return arr

def find_latest_snapshot(directory):
    fields_files = glob.glob(os.path.join(directory, "fields_*.bin"))
    if not fields_files:
        raise FileNotFoundError("No fields_XXXXX.bin files found.")
    # Extract steps
    def extract_step(fname):
        m = re.search(r'fields_(\d+)\.bin', fname)
        return int(m.group(1)) if m else None

    steps = [extract_step(f) for f in fields_files if extract_step(f) is not None]
    if not steps:
        raise FileNotFoundError("No valid snapshot files found.")

    latest_step = max(steps)
    fields_file = os.path.join(directory, f"fields_{latest_step:05}.bin")
    torsion_file = os.path.join(directory, f"torsion_{latest_step:05}.bin")

    if not os.path.exists(torsion_file):
        raise FileNotFoundError(f"Matching torsion file {torsion_file} not found.")

    return fields_file, torsion_file, latest_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize scalar field and torsion from simulation snapshots.")
    parser.add_argument("--dir", default=".", help="Directory containing snapshot files")
    parser.add_argument("--z-slice", type=int, default=0, help="Z-slice to visualize")
    parser.add_argument("--colormap-field", default="coolwarm", help="Colormap for scalar field")
    parser.add_argument("--colormap-torsion", default="plasma", help="Colormap for torsion magnitude")
    args = parser.parse_args()

    fields_file, torsion_file, step = find_latest_snapshot(args.dir)
    print(f"Loading step={step}, fields={fields_file}, torsion={torsion_file}")

    fields = load_fields(fields_file)
    torsion = load_torsion(torsion_file)

    zslice = args.z_slice

    # Extract scalar field (Φ)
    phi_slice = fields[0,0,zslice,:,:,0]

    # Compute torsion magnitude: Tmag = sqrt(sum_{a,b,c} T_{a,b,c}²)
    # After indexing: torsion[0,0,zslice,:,:,:,:,:] = (NY,NX,D,D,D)
    T_slice = torsion[0,0,zslice,:,:,:,:,:]
    Tmag = np.sqrt(np.sum(T_slice**2, axis=(2,3,4))) # (NY,NX)

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    im1 = axes[0].imshow(phi_slice, origin='lower', cmap=args.colormap_field, aspect='equal')
    axes[0].set_title("Scalar Field (Φ)")
    axes[0].set_xlabel("X index")
    axes[0].set_ylabel("Y index")
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Φ (dimensionless)")

    im2 = axes[1].imshow(Tmag, origin='lower', cmap=args.colormap_torsion, aspect='equal')
    axes[1].set_title("Torsion Magnitude")
    axes[1].set_xlabel("X index")
    axes[1].set_ylabel("Y index")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Torsion Magnitude (arb. units)")

    plt.suptitle(f"z-slice={zslice}, step={step}", fontsize=14)
    plt.tight_layout()
    plt.show()
