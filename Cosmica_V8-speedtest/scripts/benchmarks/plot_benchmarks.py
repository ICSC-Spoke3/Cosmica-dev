from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

V6 = np.array([2*60 + 19, 2*60 + 20, 2*60 + 24, 2*60 + 33, 2*60 + 35])

if __name__ == "__main__":
    directory = Path(__file__).parent
    data = np.loadtxt(directory / 'nk0_times.csv', delimiter=',')
    fig, axs = plt.subplots(2, 2, figsize=(11.25, 10), tight_layout=True)

    n_params = data[:, 0]
    V6_times = data[:, 0] * V6.mean()
    V8_times = data[:, 1:].mean(axis=1)



    ax1, ax2, ax3, ax4 = axs.flatten()

    ax1.set_title('Runtimes')
    ax1.set_xlabel('# of parametrizations')
    ax1.set_ylabel('Execution time (s)')
    ax1.plot(n_params, V8_times, label='V8 (latest)')
    ax1.plot(n_params, V6_times, '--', label='V6 (theoretical)')
    ax1.legend()

    ax2.set_title('Runtimes (y-log)')
    ax2.set_xlabel('# of parametrizations')
    ax2.set_ylabel('log(Execution time) (s)')
    ax2.set_yscale('log')
    ax2.plot(n_params, V8_times, label='V8 (latest)')
    ax2.plot(n_params, V6_times, '--', label='V6 (theoretical)')
    ax2.legend()

    ax3.set_title('Runtimes (log-log)')
    ax3.set_xlabel('log(# of parametrizations)')
    ax3.set_ylabel('log(Execution time) (s)')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.plot(n_params, V8_times, label='V8 (latest)')
    ax3.plot(n_params, V6_times, '--', label='V6 (theoretical)')
    ax3.legend()

    ax4.set_title('Speedup')
    ax4.set_xlabel('# of parametrizations')
    ax4.set_ylabel('Time Ratio [(V6 time) / (V8 time)]')
    ax4.plot(n_params, V6_times / V8_times, c='green')
    print(V6_times / V8_times)

    for ax in axs.flat:
        ax.grid()

    plt.savefig(directory / 'tmp.png')
    plt.show()
