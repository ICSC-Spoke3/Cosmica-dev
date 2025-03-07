from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

V6 = np.array([2*60 + 19, 2*60 + 20, 2*60 + 24, 2*60 + 33, 2*60 + 35])

if __name__ == "__main__":
    directory = Path(__file__).parent
    data = np.loadtxt(directory / 'nk0_times.csv', delimiter=',')
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    ax1, ax2 = axs.flatten()

    ax1.set_title('Runtimes')
    ax1.set_xlabel('# of parametrizations')
    ax1.set_ylabel('Time (s)')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid()
    ax1.plot(data[:,0], data[:, 1:].mean(axis=1))
    ax1.plot(data[:,0], data[:, 0] * V6.mean())

    ax2.set_title('Speedup')
    ax2.set_xlabel('# of parametrizations')
    ax2.set_ylabel('Time Ratio (V6 / V8)')
    ax2.grid()
    ax2.plot(data[:, 0], data[:, 0] * V6.mean() / data[:, 1:].mean(axis=1))
    # plt.savefig(directory / 'tmp.png')
    plt.show()
