# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# V6 = np.array([2*60 + 19, 2*60 + 20, 2*60 + 24, 2*60 + 33, 2*60 + 35])

# if __name__ == "__main__":
#     directory = Path(__file__).parent
#     data = np.loadtxt(directory / 'nk0_times.csv', delimiter=',')
#     fig, axs = plt.subplots(1, 2, figsize=(20, 5))
#     ax1, ax2 = axs.flatten()

#     ax1.set_title('Runtimes')
#     ax1.set_xlabel('# of parametrizations')
#     ax1.set_ylabel('Time (s)')
#     ax1.set_yscale('log')
#     ax1.set_xscale('log')
#     ax1.grid()
#     ax1.plot(data[:,0], data[:, 1:].mean(axis=1))
#     ax1.plot(data[:,0], data[:, 0] * V6.mean())

#     ax2.set_title('Speedup')
#     ax2.set_xlabel('# of parametrizations')
#     ax2.set_ylabel('Time Ratio (V6 / V8)')
#     ax2.grid()
#     ax2.plot(data[:, 0], data[:, 0] * V6.mean() / data[:, 1:].mean(axis=1))
#     plt.savefig(directory / 'tmp.png')
#     plt.show()

#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. Matplotlib config (no LaTeX)…
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 5,
})

# 2. Your data…
HERE = Path(__file__).resolve().parent
data = np.loadtxt(HERE / "nk0_times.csv", delimiter=",")
n_param  = data[:, 0]
v8_times = data[:, 1:]
v8_mean = v8_times.mean(axis=1)
v8_std  = v8_times.std(axis=1)
V6 = np.array([2*60 + 19, 2*60 + 20, 2*60 + 24, 2*60 + 33, 2*60 + 35])
v6_line = n_param * V6.mean()
speedup = v6_line / v8_mean

# → define a little custom palette:
c_v8     = "#1f77b4"  # muted blue
c_band   = "#aec7e8"  # light blue
c_v6     = "#ff7f0e"  # orange
c_speed  = "#2ca02c"  # green
c_parity = "#7f7f7f"  # gray

# 3. Plot
fig, (ax_runtime, ax_speedup) = plt.subplots(
    1, 2, figsize=(13,6),
    constrained_layout=True,
    gridspec_kw={
        'width_ratios': [2, 2],
        'wspace': 0.05,
        'hspace': 0.2,
    }
)

# Runtime panel
ax_runtime.set_title('Runtimes')
ax_runtime.set_xlabel('Number of parametrizations')
ax_runtime.set_ylabel('Wall-clock time (s)')
ax_runtime.set_xscale('log')
ax_runtime.set_yscale('log')
ax_runtime.minorticks_on()
ax_runtime.grid(which='both', ls=':', lw=0.4)

ax_runtime.fill_between(
    n_param,
    v8_mean - v8_std,
    v8_mean + v8_std,
    color=c_band,
    alpha=0.5,
    label='V8 mean ± 1 SD'
)
ax_runtime.plot(
    n_param,
    v8_mean,
    marker='o',
    color=c_v8,
    label='V8 mean'
)
ax_runtime.plot(
    n_param,
    v6_line,
    ls='--',
    color=c_v6,
    label='V6'
)
ax_runtime.legend(loc='upper left')

# Speed-up panel
ax_speedup.set_title('Speed-up')
ax_speedup.set_xlabel('Number of parametrizations')
ax_speedup.set_ylabel('Speed-up (V6 / V8)')
ax_speedup.set_xscale('log')
ax_speedup.minorticks_on()
ax_speedup.grid(which='both', ls=':', lw=0.4)

ax_speedup.plot(
    n_param,
    speedup,
    marker='s',
    color=c_speed,
    label='Observed speed-up'
)
ax_speedup.axhline(1, ls='--', lw=0.6, color=c_parity)
ax_speedup.legend(loc='upper left')

# Save & show
out = HERE / "nk0_runtime_speedup.pdf"
fig.savefig(out, dpi=600, bbox_inches="tight")
out = HERE / "nk0_runtime_speedup.png"
fig.savefig(out, dpi=600, bbox_inches="tight")
print(f"Figure written to {out.relative_to(HERE)}")
plt.show()
