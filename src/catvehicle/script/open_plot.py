#!/usr/bin/env python3


import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# SCROLL ZOOM HELPER
def add_scroll_zoom(fig, scale=1.3):
    def on_scroll(event):
        ax = event.inaxes
        if ax is None:
            return
        factor = 1 / scale if event.button == 'up' else scale
        ax.set_xlim([event.xdata - (event.xdata - ax.get_xlim()[0]) * factor,
                     event.xdata + (ax.get_xlim()[1] - event.xdata) * factor])
        ax.set_ylim([event.ydata - (event.ydata - ax.get_ylim()[0]) * factor,
                     event.ydata + (ax.get_ylim()[1] - event.ydata) * factor])
        fig.canvas.draw_idle()

    def on_dblclick(event):
        ax = event.inaxes
        if ax is None:
            return
        ax.autoscale()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event',
                           lambda e: on_dblclick(e) if e.dblclick else None)


# PATHS
CSV_PATH = '/home/student/lane_detection_data_mpc.csv'
PKL_PATH = '/home/student/mpc_plot.pkl'
SAVE_DIR = '/home/student/'
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})


# LOAD DATA
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

t     = (df['Timestamp'] - df['Timestamp'].iloc[0]).to_numpy()
steer = df['Steering_Rad'].to_numpy()

# Steering rate (rad/s)
dt = np.diff(t, prepend=t[0])
dt[dt == 0] = np.nan
steer_rate = np.diff(steer, prepend=steer[0]) / dt

# Rolling RMS of steering rate
steer_rms = (
    pd.Series(steer_rate)
    .rolling(window=10, center=True)
    .apply(lambda x: np.sqrt(np.nanmean(x**2)), raw=True)
    .to_numpy()
)


# Steering Angle vs Time
fig1 = plt.figure(figsize=(10, 4))
plt.plot(t, steer, linewidth=1.2)
plt.axhline(0.0, color='k', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel('Steering Angle [rad]')
plt.title('Steering Angle vs Time')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'steering_angle_vs_time.png'), dpi=300)
add_scroll_zoom(fig1)
print("[1/3] Steering plot saved.")


# Steering Smoothness (RMS)
fig2 = plt.figure(figsize=(10, 4))
plt.plot(t, steer_rms, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('RMS Steering Rate [rad/s]')
plt.title('Steering Smoothness (Rolling RMS)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'steering_smoothness.png'), dpi=300)
add_scroll_zoom(fig2)
print("[2/3] Smoothness plot saved.")


# Vehicle Path vs Reference
with open(PKL_PATH, 'rb') as f:
    fig3 = pickle.load(f)

fig3.set_size_inches(6, 8)
plt.figure(fig3.number)
plt.title('Vehicle Path vs Reference Trajectory')
plt.savefig(os.path.join(SAVE_DIR, 'path_vs_reference.png'), dpi=300, bbox_inches='tight')
add_scroll_zoom(fig3)
print("[3/3] Path plot saved.")

print("\n[DONE] Plots saved to:", SAVE_DIR)
print("Scroll to zoom | Double-click to reset")
plt.show()