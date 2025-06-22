"""
    A library of standard and custom metric functions for evaluating cosmic ray propagation simulations
    against experimental data. Each metric is a callable suitable for use in optimization routines.

    Metrics compare simulation output (ModulationResult) to reference data (ExperimentalData).
"""

import numpy as np

def rmse(result, experimental_data):
    """
        Root Mean Squared Error between simulation and experiment.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: The RMSE value.
    """
    return float(np.sqrt(np.mean((result.flux - experimental_data.flux) ** 2)))


def mae(result, experimental_data):
    """
        Mean Absolute Error between simulation and experiment.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: The mean absolute error.
    """
    return float(np.mean(np.abs(result.flux - experimental_data.flux)))


def mean_relative_error(result, experimental_data):
    """
        Mean Relative Error between simulation and experiment.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: The mean relative error.
    """
    rel_error = np.abs((result.flux - experimental_data.flux) / experimental_data.flux)
    return float(np.mean(rel_error))


def max_abs_error(result, experimental_data):
    """
        Maximum Absolute Error between simulation and experiment.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: The maximum absolute error.
    """
    return float(np.max(np.abs(result.flux - experimental_data.flux)))


def msle(result, experimental_data):
    """
        Mean Squared Logarithmic Error between simulation and experiment.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: The mean squared logarithmic error.
    """
    flux_pred = np.maximum(result.flux, 0)
    flux_true = np.maximum(experimental_data.flux, 0)
    return float(np.mean((np.log1p(flux_pred) - np.log1p(flux_true)) ** 2))


def log_rmse(result, experimental_data):
    """
        Logarithmic Root Mean Squared Error between simulation and experiment.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: The logarithmic RMSE value.
    """
    flux_pred = np.maximum(result.flux, 1e-10)
    flux_true = np.maximum(experimental_data.flux, 1e-10)
    return float(np.sqrt(np.mean((np.log10(flux_pred) - np.log10(flux_true)) ** 2)))


def weighted_rmse(result, experimental_data, rigidities=None):
    """
        Weighted Root Mean Squared Error between simulation and experiment,
        using rigidities as weights.    
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
            rigidities (array-like, optional): Rigidities to use as weights. If None, uses experimental data rigidities.
        Returns:
            float: The weighted RMSE value.
    """
    if rigidities is None:
        rigidities = getattr(experimental_data, "rig_flux", None)
        if rigidities is not None:
            rigidities = rigidities.rigidity
        else:
            rigidities = np.ones_like(result.flux)
    weights = rigidities / np.sum(rigidities)
    return float(np.sqrt(np.sum(weights * (result.flux - experimental_data.flux) ** 2)))



def frac_within_percent(result, experimental_data, percent=10):
    """
        Fraction of fluxes within a given percentage of experimental data.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
            percent (float): Percentage threshold to consider.
        Returns:
            float: Fraction of fluxes within the specified percentage.
    """
    perc_err = 100 * np.abs((result.flux - experimental_data.flux) / experimental_data.flux)
    return float(np.mean(perc_err < percent))


def mean_flux_ratio(result, experimental_data):
    """
        Mean flux ratio between simulation and experimental data.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: Mean ratio of simulation flux to experimental flux.
    """
    flux_true = np.maximum(experimental_data.flux, 1e-10)
    return float(np.mean(result.flux / flux_true))


def pearson_corr(result, experimental_data):
    """
        Pearson correlation coefficient between simulation and experimental fluxes.
        Args:
            result (ModulationResult): Simulation output containing flux data.
            experimental_data (ExperimentalData): Reference data containing flux values.
        Returns:
            float: Pearson correlation coefficient.
    """
    return float(np.corrcoef(result.flux, experimental_data.flux)[0, 1])