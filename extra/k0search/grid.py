import numpy as np

from physics_utils import eval_k0


def k0_grid_from_estimate(t: int, nk0: int, h_par: np.ndarray, q=+1, debug=False):
    """
    Generate the grid of k0 values
    :param t: the time step
    :param nk0: the number of k0 values
    :param h_par: the heliospheric parameters
    :param q: the sign of particle charge
    :param debug: if True, more verbose output will be printed
    :return: k0grid (the grid of k0 values), k0 (estimated k0), k0err (estimated k0 error)
    """

    # Find first row that contains t: c1 <= t < c2
    i = np.argmax((h_par[:, 0] <= t) & (t < h_par[:, 1]))

    # Extract some parameters
    ssn, tilt_l, polarity, solar_phase, nmcr = h_par[i, [2, 4, 7, 8, 11]]
    # Compute high activity period (if next 15 rows are high activity)
    is_high_activity_period = (np.average([float(tilt) for tilt in h_par[i:i + 15, 4]])) > 50

    # Estimate (fit) reference k0 value and corresponding error
    k0, k0err = eval_k0(is_high_activity_period, polarity, q, solar_phase, tilt_l, nmcr, ssn)
    if debug:
        print(f"IsHighActivityPeriod {is_high_activity_period}")
        print(f"K0 {k0} +- {k0err * k0}")

    # Generate a grid around the error with 4 x standard deviations range
    k0grid = np.linspace(k0 - 4 * k0err * k0, k0 + 4 * k0err * k0, num=nk0, endpoint=True)
    return k0grid, k0, k0err
