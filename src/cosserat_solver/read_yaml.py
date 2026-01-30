from __future__ import annotations

import yaml

import cosserat_solver.consts as consts


def read(file_path: str):
    """
    Read a YAML file and return its contents as a dictionary.

    Parameters:
    file_path: str
        The path to the YAML file.

    Returns:
    material_params: dict
        A dictionary containing the material parameters:
        - 'rho': Density
        - 'lam': Lamé's first parameter
        - 'mu': Shear modulus
        - 'nu': Cosserat couple modulus
        - 'J': Micro-inertia
        - 'lam_c': Cosserat Lamé's first parameter
        - 'mu_c': Cosserat shear modulus
        - 'nu_c': Cosserat couple modulus

    source_params: dict
        A dictionary containing the source parameters:
        - 'f0': Dominant frequency of the Ricker wavelet in Hz.
        - 'f': Displacement ratio.
        - 'fc': Rotation ratio.
        - 'angle': Source angle in degrees.

    ft_params: dict
        A dictionary containing the parameters for the Fourier transform. It should contain the following keys:
        - 'dt': The time step size for the output trace.
        - 'N': The number of time samples for the output trace.
        - 'extension_factor': An integer specifying the factor for extending the window.

    digits_precision: int
        The number of digits of precision for intermediate mpmath calculations.

    seismogram_locations: list of tuples
        A list of (x, z) tuples specifying the locations of seismograms.
    """
    with open(file_path) as file:
        data = yaml.safe_load(file)

    material_params = data.get("material_params", {})
    source_params = data.get("source_params", {})
    ft_params = data.get("ft_params", {})
    digits_precision = data.get("digits_precision", consts.COMPUTE_PRECISION)
    seismogram_locations = data.get("seismogram_locations", [])

    return (
        material_params,
        source_params,
        ft_params,
        digits_precision,
        seismogram_locations,
    )
