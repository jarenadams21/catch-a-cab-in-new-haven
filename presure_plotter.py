import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
data = pd.read_csv("laser_experiment_results.csv")

time = data["time(s)"]
raw_intensity = data["raw_intensity"]
transmitted = data["transmitted_intensity"]
pol_angle = data["polarization_angle(deg)"]
gw_metric = data["gw_metric_factor"]
epsilon = data["epsilon(J/m^3)"]
pressure = data["pressure(J/m^3)"]
spin_density = data["spin_density"]

# Plot transmitted intensity over time
plt.figure(figsize=(8,6))
plt.plot(time, transmitted, label="Transmitted Intensity (after analyzer)", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Intensity (arb. units)")
plt.title("Transmitted Intensity Over Time (EoS + Spin-Torsion)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("transmitted_intensity.png", dpi=150)

# Plot polarization angle over time
plt.figure(figsize=(8,6))
plt.plot(time, pol_angle, label="Polarization Angle", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Polarization Angle (degrees)")
plt.title("Polarization Angle Evolution")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("polarization_angle.png", dpi=150)

# Plot GW metric factor
plt.figure(figsize=(8,6))
plt.plot(time, gw_metric, label="GW Metric Factor", color='red')
plt.xlabel("Time (s)")
plt.ylabel("GW Metric Factor")
plt.title("Gravitational Wave Metric Perturbation Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gw_metric_factor.png", dpi=150)

# Plot EoS-related fields: epsilon, pressure, and spin_density
fig, ax1 = plt.subplots(figsize=(8,6))

color = 'tab:blue'
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Energy Density (J/m^3)", color=color)
ax1.plot(time, epsilon, color=color, label="Energy Density")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel("Pressure (J/m^3)", color=color)
ax2.plot(time, pressure, color=color, label="Pressure")
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.title("EoS Evolution: Energy Density and Pressure")
plt.savefig("eos_evolution.png", dpi=150)

plt.figure(figsize=(8,6))
plt.plot(time, spin_density, label="Neutrino Spin Density", color='purple')
plt.xlabel("Time (s)")
plt.ylabel("Spin Density (arb. units)")
plt.title("Neutrino Spin Density Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("spin_density_evolution.png", dpi=150)

plt.show()