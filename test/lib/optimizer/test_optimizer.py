from typing import List, Union

import numpy as np

from matplotlib import pyplot as plt

import nevergrad as ng

from test.lib.fstpso_ask_tell.fstpso import FuzzyPSO
from test.lib.optimizer.optimizer import Optimizer, OptimizerQueue


def real_function(x):
    return np.sum(x)


def dummy_metric(real_data, x):
    return real_function(x) # Optional: slight noise

# Dummy procedure

def dummy_procedure(params: List[List[Union[float, int]]]) -> List[List[float]]:
    """
    Squares each element in a 2D list of real-valued vectors.

    :param params: A list of 1D vectors (e.g., [[x1, x2], [y1, y2]])
    :return: A list of 1D vectors with squared values.
    """
    return params

def test_optimizer():


    # Testing Using fuzzy PSO


    dimensions = 1
    optimizer_fuzzy_pso = FuzzyPSO()
    optimizer_fuzzy_pso.set_search_space([[-5,5]]*dimensions)
    optimizer_fuzzy_pso.InitCreateParticles(1000, dimensions)


    optimizer_fuzzy_pso._prepare_for_optimization(max_iter=100,max_iter_without_new_global_best=50,  max_FEs = None)

    optimizer_FUZZYPSO = Optimizer(optimizer=optimizer_fuzzy_pso,name="Fuzzy-PSO",eval_metric=dummy_metric,
                                   save_results_ended=False,external_stopping_criteria=optimizer_fuzzy_pso.TerminationCriterion)

    vector_param = ng.p.Array(shape=(1,)).set_bounds(-5, 5)
    parametrization = ng.p.Instrumentation(vector_param)
    optim = ng.optimizers.registry["CMA"](
        parametrization=parametrization,
        budget=1000,
        num_workers=1000
    )

    optim_DE = ng.optimizers.registry["DE"](
        parametrization=parametrization,
        budget=1000,
        num_workers=1000
    )
    optimizer_CMA = Optimizer(optimizer=optim,name="CMA",min_increment=1e-4,max_iterations=1000,eval_metric=dummy_metric,save_results_ended=False,max_size_best_candidates=10,budget=1000)
    optimizer_DE = Optimizer(optimizer=optim_DE,name="DE",min_increment=1e-4,max_iterations=1000,eval_metric=dummy_metric,save_results_ended=False,max_size_best_candidates=10,budget=1000)


    queue = OptimizerQueue([optimizer_CMA,optimizer_DE,optimizer_FUZZYPSO])
    queue.start(real_data=[0], epochs=1, procedure=dummy_procedure,initial_value_parameter=[0.2])

    plotInfo(optimizer=optimizer_CMA)
    plotInfo(optimizer=optimizer_DE)
    plotInfo(optimizer=optimizer_FUZZYPSO)



def plotInfo(optimizer:Optimizer):
    x_vals = np.linspace(-1, 5, 100).reshape(-1, 1)  # shape becomes (100, 1)
    y_vals = []
    for x in x_vals:
        y_vals.append(real_function(x))

    # Suppose optimizer_steps is a list of (x, loss) tuples you collected
    optimizer_steps = []
    for step, loss in optimizer.candidate_seen:
        if hasattr(step, 'value'):  # likely Instrumentation
            val = step.value[0][0][0]
        elif isinstance(step, (tuple, list)):
            val = float(step[0])
        else:
            val = list(step)[0]
        optimizer_steps.append((val, loss))


    # Extract steps and corresponding losses
    steps_x = [x for x, _ in optimizer_steps]
    steps_y = [real_function(x) for x in steps_x]

    plt.figure(figsize=(8,6))
    plt.plot(x_vals, y_vals, label='Real Function', color='blue')
    plt.scatter(steps_x, steps_y, color='red', label='Optimizer Steps')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('Optimizer Steps vs Real Function for Optimizer' + optimizer.get_name() )
    plt.grid(True)
    plt.show()

    # Extract losses over time (y-axis) in the order they were evaluated
    step_indices = steps_x
    step_losses = [loss for x, loss in optimizer.candidate_seen]  # y-axis = loss at that step

    plt.figure(figsize=(8, 6))
    plt.scatter(step_indices, step_losses, color='green', label='Fitness per Step')
    plt.xlabel('Step')
    plt.ylabel('Fitness (f(x))')
    plt.title('Fitness over Optimization Steps' + optimizer.get_name())
    plt.grid(True)
    plt.legend()
    plt.show()





if __name__ == "__main__":
    test_optimizer()