"""
Magnetic field simulation using magpylib based on configuration parameters.
Reads reference positions from ref_pos.csv, computes magnetic field from defined cylindrical magnets,
and saves reference and noisy magnetometer data, matching the style of ref_mag.csv and mag-0.csv.
Also generates plots of magnetic field strength and field vectors.
"""
import os
import numpy as np
import pandas as pd
import magpylib as magpy
import matplotlib.pyplot as plt

# Configuration parameters
config = {
    'magnets': [
        {'position': [-0.08,  0.2, -0.3], 'moment_direction': [0, 0, 1], 'radius_mm': 10, 'height_mm': 5, 'magnetization_strength': 1e6},
        {'position': [ 0.08,  0.2, -0.3], 'moment_direction': [0, 0, 1], 'radius_mm': 10, 'height_mm': 5, 'magnetization_strength': 1e6},
        {'position': [ 0.05, 0, -0.3], 'moment_direction': [0, 0, 1], 'radius_mm': 10, 'height_mm': 5, 'magnetization_strength': 1e6},
        {'position': [-0.05, 0, -0.3], 'moment_direction': [0, 0, 1], 'radius_mm': 10, 'height_mm': 5, 'magnetization_strength': 1e6},
        {'position': [ 0.0,  -0.02,  -0.3], 'moment_direction': [0, 0, 1], 'radius_mm': 10, 'height_mm': 5, 'magnetization_strength': 1e6},
    ],
    'measurement_noise_std': [0.1, 0.1, 0.1],  # in microTesla
    'hard_iron_bias': [0.05, -0.02, 0.1],       # in microTesla
}

# Paths
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets', 'gnss_ins_sim_test'))
ref_pos_file = os.path.join(dataset_dir, 'ref_pos.csv')
ref_mag_file = os.path.join(dataset_dir, 'ref_mag.csv')
mag0_file    = os.path.join(dataset_dir, 'mag-0.csv')

# Load reference positions (meters)
pos_df = pd.read_csv(ref_pos_file)
pos = pos_df.iloc[:, :3].values  # assume columns ref_pos_x, ref_pos_y, ref_pos_z

# Initialize magnets
grid = []
for m in config['magnets']:
    mag = magpy.magnet.Cylinder(
        magnetization=np.array(m['moment_direction']) * m['magnetization_strength'],
        dimension=(m['radius_mm'] / 1000.0, m['height_mm'] / 1000.0)
    ).move(m['position'])
    grid.append(mag)

# Compute magnetic field at each position (Tesla)
B_fields = np.array([sum([mag.getB(p) for mag in grid]) for p in pos])
# Convert to microTesla
B_uT = B_fields * 1e6

# Save reference magnetic field
ref_df = pd.DataFrame(B_uT, columns=['ref_mag_x (uT)', 'ref_mag_y (uT)', 'ref_mag_z (uT)'])
ref_df.to_csv(ref_mag_file, index=False)

# Apply noise and hard iron bias
dnoise = np.random.randn(*B_uT.shape) * np.array(config['measurement_noise_std'])
B_meas = B_uT + dnoise + np.array(config['hard_iron_bias'])

# Save measured magnetic field
meas_df = pd.DataFrame(B_meas, columns=['mag_x (uT)', 'mag_y (uT)', 'mag_z (uT)'])
meas_df.to_csv(mag0_file, index=False)

# Plot magnetic field strength over samples
strength_ref = np.linalg.norm(B_uT, axis=1)
strength_meas = np.linalg.norm(B_meas, axis=1)
plt.figure(figsize=(8, 4))
plt.plot(strength_ref, label='Reference', color='blue')
plt.plot(strength_meas, label='Measured', color='orange', alpha=0.6)
plt.title('Magnetic Field Strength over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Field Strength (uT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dataset_dir, 'magnetic_field_strength.png'))


# 绘制 Bx、By、Bz 三个分量随样本变化的组合图
samples = np.arange(B_meas.shape[0])
plt.figure(figsize=(8, 4))
plt.plot(samples, B_meas[:, 0], label='Bx (uT)', color='red')
plt.plot(samples, B_meas[:, 1], label='By (uT)', color='green')
plt.plot(samples, B_meas[:, 2], label='Bz (uT)', color='blue')
plt.title('Magnetic Field Components over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Field Strength (uT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dataset_dir, 'mag_components.png'))

# 绘制磁场矢量在 xy 平面上的轨迹
plt.figure(figsize=(6, 6))
plt.plot(pos[:, 0], pos[:, 1], 'k-', label='Sensor Trajectory')
# 标注磁铁位置
xs = [m['position'][0] for m in config['magnets']]
ys = [m['position'][1] for m in config['magnets']]
plt.scatter(xs, ys, color='red', s=60, label='Magnet Positions')
plt.title('Sensor Trajectory and Magnet Positions (xy plane)')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(dataset_dir, 'trajectory_xy.png'))