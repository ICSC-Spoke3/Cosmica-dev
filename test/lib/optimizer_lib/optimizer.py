"""

"""
import random
from copy import deepcopy
from datetime import time
from pathlib import Path
import bisect
import os

import nevergrad
import numpy as np
import pandas as pd
from typing import Callable, Optional, List, Tuple
import time

from astropy.stats import knuth_bin_width


class OptimizerQueue:
    def __init__(self, optimizers_list: List[object]):
        self.optimizer_queue = []
        self.completed_list = []
        self.optimizer_to_test = []
        # optimizations get information from all candidates seen from each other
        self.use_combinate_information = bool

        for optimizer in optimizers_list:
            if hasattr(optimizer, "ask") and callable(getattr(optimizer, "ask")) and \
                    hasattr(optimizer, "tell") and callable(getattr(optimizer, "tell")):
                self.optimizer_to_test.append(optimizer)
            else:
                print(f"{optimizer} skipped, 'ask' or 'tell' method not found")

    def start(self,
              real_data,
              epochs: int,
              procedure: Callable = None,
              initial_value_parameter: List[float] = [0.0],
              generate_random_seed_each_epoch: bool = False,
              random_seed: Optional[int] = None,
              experimental_data_str : str = "",
              n_part = 4096,
              **kwargs):


        if procedure is None:
            raise ValueError("Procedure function must be provided.")

        k_validation = kwargs.get("k_validation", 1)
        k_validation_func = kwargs.get("k_validation_func", lambda values: np.mean(values, axis=0))


        for optimizer in self.optimizer_to_test:
            self.optimizer_queue.append(optimizer)


        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1} ---")

            print(f"\n--- Generating Random Seed ---")
            if generate_random_seed_each_epoch:
                #TODO change
                random_seed = int(time.time())
                random.seed(random_seed)

            print(f"\n--- Used Seed {random_seed} ---")

            # Start Epoch

            # initial value parameter is a list of default value to test
            # each value to test rappresent a n-dimensional vector in the space
            dummy_inputs = initial_value_parameter

            print(f"\n--- initial Value {dummy_inputs} ---")


            print(f"\n--- Results from {getattr(procedure, '__name__', str(procedure))} ---")


            for optimizer in self.optimizer_queue[:]:
                print(f"\n--- starting optimizer {optimizer.name} ---")

                optimizer.run(seed=random_seed)


                start_time_epoch = time.time()

                fitness = self.apply_k_validation(dummy_inputs, k_validation, k_validation_func, optimizer, procedure,
                                                  random_seed, real_data)

                optimizer.times_elapsed_per_epoch.append(time.time() - start_time_epoch)


                print(f"\n--- Telling :  {dummy_inputs} , {fitness}   ---")

                optimizer.tell(known_solutions=dummy_inputs,fitness= fitness,is_starting_point = True)


            for optimizer in self.optimizer_queue[:]:
                print(str(optimizer) + " turn")

                while not optimizer.check_stopping_criteria():
                    print(f"\n--- Getting candidates   ---")

                    candidates = optimizer.ask(n=optimizer.budget)
                    start_time_epoch = time.time()

                    fitness = self.apply_k_validation(candidates, k_validation, k_validation_func, optimizer, procedure,
                                              random_seed, real_data)

                    optimizer.times_elapsed_per_epoch.append(time.time() - start_time_epoch)

                    print(f"\n--- Candidates :  {candidates}    ---")

                    print(f"\n--- Telling :  {optimizer.last_asked} , {fitness}   ---")

                    optimizer.tell(fitness)

                print(f"\n--- Optimizer {optimizer.name} reached stopping criteria   ---")

                self.completed_list.append(optimizer)
                self.optimizer_queue.remove(optimizer)

                print(f"\n--- Optimizer {optimizer.name} saving results   ---")

                if optimizer.save_results_ended:
                    optimizer.save_results(experimental_data_str,default=True,n_part=n_part)

            print(f"Active: {len(self.optimizer_queue)}, To convergence: {len(self.completed_list)}")

    def apply_k_validation(self, dummy_inputs, k_validation, k_validation_func, optimizer, procedure, random_seed,
                           real_data):
        all_fitness = []
        for k_val in range(k_validation):
            seed_step = None if random_seed is None else int(random_seed + k_val)
            results = procedure(dummy_inputs, seed_step)
            fitness = optimizer.evaluate_fitness(real_data, results)
            all_fitness.append(fitness)
        fitness_mean = k_validation_func(all_fitness)
        return fitness_mean


def extract_value(solution)->List:
    if hasattr(solution, 'value'):
        return solution.value
    elif hasattr(solution, 'X'):
        return deepcopy(solution.X)
    else:
        return solution


def set_param_value(param_obj, value):
    """
    Sets the parameter value to param_obj.X if it exists,
    otherwise sets it to param_obj.value.
    """
    if hasattr(param_obj, 'X'):
        param_obj.X = value
    else:
        param_obj.value = value


class Optimizer:

    def __init__(self,
                 optimizer: object,
                 eval_metric:Callable,
                 name: str,
                 min_increment: float = 1e-4,
                 max_iterations: Optional[int] = 100,
                 parquet_dir:Optional[str|Path] = "",
                 save_results_ended = True,
                 external_stopping_criteria: Optional[Callable[[], bool]] = None,
                 max_size_best_candidates = 10,
                 use_default_budget = True,
                 budget:Optional[int|None] = None):

        if not (hasattr(optimizer, "ask") and callable(getattr(optimizer, "ask")) and
                hasattr(optimizer, "tell") and callable(getattr(optimizer, "tell"))):
            raise ValueError("Optimizer must have 'ask' and 'tell' methods.")

        self.optimizer = optimizer
        self.name = name
        if external_stopping_criteria is not None:
            self.stopping_criteria = external_stopping_criteria
        else:
            self.stopping_criteria = StoppingCriteria(min_increment, max_iterations)

        self.max_size_best_candidates = max_size_best_candidates
        self.best_candidate = (None, float('inf'))  # (params, loss)
        self.candidate_seen = []
        self.seed = None
        self.actual_iterations = 0
        self.eval_metric = eval_metric
        self.save_results_ended = save_results_ended
        self.parquet_dir = parquet_dir
        self.last_asked = []
        self.running_time = 0
        self.best_fitness_per_epoch = []
        self.best_x_per_epoch = []
        self.times_elapsed_per_epoch = []
        self.budget = budget
        self.max_iterations = max_iterations

        if use_default_budget:
            if hasattr(self.optimizer, 'budget'):
                self.budget = self.optimizer.budget
            elif hasattr(self.optimizer, 'numberofparticles'):
                self.budget = self.optimizer.numberofparticles


    def run(self, seed: int, eval_metric: Optional[Callable] = None):
        self.best_candidate = (None, float('inf'))
        self.candidate_seen = []
        self.seed = seed
        if eval_metric is not None:
            self.eval_metric = eval_metric
        self.actual_iterations = 0
        self.running_time = time.time()


    def ask(self,use_default_population_size = True, n: int = 1) -> List:
        '''
            Ideally, nevergrad still works even with n_ask < n_populations
            but that can lead to downgrade optimization's performance.
            For non-nevergrad object, n_ask should be equal to n_population if using
            Evolutionary Strategy like Fuzzy-PSO.
            - For nevergrad compatibility, we check if optimizer has parameter value since every
            asked Parametrization is instance of Nevergrad Instrumentation class.
            - For Non-nevergrad we check if optimizer has parameter X (chosen to be standard field for Optimizer).
            we continue to ask until we get a None value

        :param n: number of parameters to aks
        :return:
        '''

        self.last_asked = []

        if use_default_population_size and not isinstance(self.optimizer, nevergrad.optimization.Optimizer):
            while (ask := self.optimizer.ask()) is not None:
                self.last_asked.append(ask)
        else:
            self.last_asked = [self.optimizer.ask() for _ in range(n)]

        values = []
        for param in self.last_asked:

            val = extract_value(solution=param)

            if isinstance(val, (float, int)):
                values.append(val)
            elif isinstance(val, (list, tuple, np.ndarray)):
                if isinstance(val,tuple):
                    # nevergrad standard
                    values.append(np.array(val[0]).flatten().tolist())
                else:
                    values.append(np.array(val).flatten().tolist())
            else:
                raise ValueError(f"Unsupported value type: {type(val)}")

        return values

    def update_internal_status(self,param_value_copy,fit):

        self.candidate_seen.append((param_value_copy, fit))
        ### extract position 1 bc is a tuple

        if fit < self.best_candidate[1]:
            self.best_candidate = (param_value_copy, fit)

    def tell(self, fitness: List[float], known_solutions: Optional[List|None] = None,is_starting_point = True):
        '''
            Basically we can tell non-present solutions to Optimizer  if so we need to spawn child with passed fit and solution's value.
            For nevergrad compatibility we check if parametrization attribute is present on optimizer since optimizer.parametrization must be
            call to spawn child.
            For Non-nevergrad Optimizer we assume that spawn_child function exists.

        :param is_starting_point:
        :param fitness: fitness value, assuming that each fitness value is passed with same parameter's index.
        :param known_solutions: passed solutions that optimizer still doesn't know.
        :return:
        '''

        best_fitness_in_epoch = np.inf
        best_x_in_epoch = 0

        if known_solutions is not None:
            if len(known_solutions) == 1:
                known_solution_items = [(deepcopy(known_solutions[0]), fitness[0])]
            else:
                known_solution_items = [(deepcopy(param), fit) for param, fit in zip(known_solutions, fitness)]

            for param_value_copy, fit in known_solution_items:

                if hasattr(self.optimizer, 'parametrization'):
                    # Nevergrad way
                    param_obj = self.optimizer.parametrization.spawn_child()
                    param_obj.value = ((param_value_copy,), {})

                elif hasattr(self.optimizer, 'spawn_child'):
                    if is_starting_point and not isinstance(self.optimizer, nevergrad.optimization.Optimizer) and len(known_solutions) == 1:
                        # set all solutions to constant initial value
                        for i in range(len(self.optimizer.Solutions)):
                            param_obj = self.optimizer.spawn_child()
                            set_param_value(param_obj=param_obj, value=deepcopy(param_value_copy))
                            self.optimizer.tell(param_obj, fit)

                        if best_fitness_in_epoch > fit:
                            best_x_in_epoch = param_value_copy
                            best_fitness_in_epoch = fit

                        self.update_internal_status(param_value_copy=param_value_copy,fit=fit)

                        continue  # avoid telling again

                    param_obj = self.optimizer.spawn_child()
                    set_param_value(param_obj=param_obj, value=param_value_copy)
                    if best_fitness_in_epoch > fit:
                        best_x_in_epoch = param_value_copy
                        best_fitness_in_epoch = fit

                    self.update_internal_status(param_value_copy=param_value_copy,fit=fit)


                else:
                    #fallback: call tell directly with value
                    if best_fitness_in_epoch > fit:
                        best_x_in_epoch = param_value_copy
                        best_fitness_in_epoch = fit

                    self.optimizer.tell(param_value_copy, fit)
                    self.update_internal_status(param_value_copy=param_value_copy,fit=fit)

                    continue  # no param_obj to tell

                self.update_internal_status(param_value_copy=param_value_copy,fit=fit)

        else:
            #Tell the optimizer the fitness for the last asked parameters
            for param_value, fit in zip(self.last_asked, fitness):

                # For Nevergrad, param_value is param object
                # For PSO_new, param_value is Particle
                param_value_copy = deepcopy(param_value)

                if best_fitness_in_epoch > fit:
                    best_x_in_epoch = param_value_copy
                    best_fitness_in_epoch = fit
                self.update_internal_status(param_value_copy=param_value_copy,fit=fit)

                if hasattr(self.optimizer, 'parametrization'):
                    self.optimizer.tell(param_value, fit)
                elif hasattr(self.optimizer, 'tell'):
                    # PSO_new expects (Particle, fitness) or (value, fitness)
                    self.optimizer.tell(param_value, fit)
                else:
                    raise Exception("Optimizer does not have tell method")


        self.best_x_per_epoch.append(best_x_in_epoch)
        self.best_fitness_per_epoch.append(best_fitness_in_epoch)

        # we dont consider initial parametrization as iteration
        if known_solutions is None:
            self.actual_iterations += 1

        print(self.actual_iterations)

    def get_name(self):
        return self.name

    def get_last_asked(self):
        return self.last_asked

    def get_last_best_asked(self):
        return self.best_last_asked

    def get_best_candidate(self):
        return self.best_candidate

    def get_iteration_number(self):
        return self.actual_iterations

    def evaluate_fitness(self, real_data: List, results: List) -> List[float]:
        return [self.eval_metric(real_data, r) for r in results]

    def check_stopping_criteria(self) ->bool:
        '''
            stopping criteria can be a Stopping criteria class that can be passed to constructor at initialization
            else can be specified the optimizer's function that must be call to check termination criteria.

        :return:
        '''
        if callable(self.stopping_criteria):
            return self.stopping_criteria()
        elif type(self.stopping_criteria) is StoppingCriteria:
            return self.stopping_criteria.check(self, self.actual_iterations)
        else:
            raise Exception("None valid stopping criteria passed.")

    def save_results(self,experimental_data_path = "",default: bool = True,n_part = 4096):
        if not default:
            return

        os.makedirs(os.path.dirname(self.parquet_dir), exist_ok=True)

        df_new = pd.DataFrame([self.to_dict(experimental_data_path,n_part = n_part)])
        if os.path.exists(self.parquet_dir):
            df_existing = pd.read_parquet(self.parquet_dir)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_parquet(self.parquet_dir, index=False, engine="pyarrow")

    def to_dict(self,experimental_data_path = "",n_part = 4096,k_validation=2,) -> dict:


        best_x,best_loss = self.best_candidate
        best_x = extract_value(best_x)

        steps = []
        evaluated_loss = []
        best_x_each_epoch = []

        for x, loss in self.candidate_seen:
            steps.append(extract_value(x))
            evaluated_loss.append(loss)

        for x in self.best_x_per_epoch:
            best_x_each_epoch.append(extract_value(x))

        print(best_x_each_epoch)

        return {
            "best_x": best_x,
            "best_loss": best_loss,
            "steps": steps,
            "corr_loss": evaluated_loss,
            "seed": self.seed,
            "time_elapsed": time.time() - self.running_time ,
            "algorithm": self.name,
            "iterations": self.actual_iterations,
            "n_part": n_part,
            "experimental_data_path": experimental_data_path,
            "eval_metric": getattr(self.eval_metric, '__name__', str(self.eval_metric)),
            "best_per_epoch": self.best_fitness_per_epoch,
            "best_k0_per_epoch":best_x_each_epoch,
            "max_iterations":self.max_iterations,
            "population" : self.budget,
            "k_validation":k_validation,
            "times_elapsed_per_epoch" : self.times_elapsed_per_epoch,
            "version":"v2"
        }

    def __str__(self):
        return self.name




# Base class for stopping criteria check
# TODO implement with new logic for early stopping
class StoppingCriteria:

    def __init__(self, min_increment: float = 1e-4, max_iterations: int = 100):
        self.set_min_increment = min_increment is not None
        self.min_increment = min_increment
        self.set_iteration_limit = max_iterations is not None
        self.max_iterations = max_iterations

    def check(self, optimizer: Optimizer, iteration_counter: int) -> bool:

        print(f"{self.set_iteration_limit} : {iteration_counter} / {self.max_iterations}")

        if self.set_iteration_limit and iteration_counter >= self.max_iterations:
            return True

        if optimizer.get_iteration_number() > 1:
            mean_fit = np.mean([fit for _, fit in optimizer.get_last_best_asked()])
            best_fit = optimizer.get_best_candidate()[1]

            if np.abs(mean_fit - best_fit) < self.min_increment:
                return True

        return False




def OptimizerAskTellBase():
    def ask():
        pass
    def tell():
        pass
    def spawn_child():
        pass
