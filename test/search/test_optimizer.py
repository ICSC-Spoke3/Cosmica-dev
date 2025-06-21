import nevergrad as ng
import numpy as np

import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    population_size = 4
    lr = ng.p.Scalar(lower=10, upper=30)
    parametrization = ng.p.Instrumentation(lr)

    bench_func = lambda x: 10 + x**2 - 10 * np.cos(2 * np.pi * x)
    loss_f = lambda f: f

    names = ["CMA"]

    x_points = []
    y_points = []

    for name in names:
        optim = ng.optimizers.registry[name](
            parametrization=parametrization,
            budget=population_size * 10,
            num_workers=population_size
        )

        for iteration in range(10):
            k0_list = [optim.ask() for _ in range(population_size)]

            fitness = []
            for k0 in k0_list:
                x_val = k0.value[0][0]
                y_val = bench_func(x_val)
                fitness.append(loss_f(y_val))

                x_points.append(x_val)
                y_points.append(y_val)

            for k0, fit in zip(k0_list, fitness):
                optim.tell(k0, fit)

    # Get best candidate
    best = optim.provide_recommendation()
    best_x = best.value[0][0]
    best_y = bench_func(best_x)

    # Plot function
    X = np.linspace(10, 30, 1000)
    Y = bench_func(X)
    plt.plot(X, Y, label="Rastrigin Function", linewidth=2)

    # Plot evaluated points
    plt.scatter(x_points, y_points, c='orange', label="Evaluated Points")

    # Plot best point
    plt.scatter([best_x], [best_y], c='red', label=f"Best: x={best_x:.3f}", s=100, marker='X')

    plt.title("Rastrigin Function Optimization with CMA")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
