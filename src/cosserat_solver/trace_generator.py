from __future__ import annotations

import os

import numpy as np

import cosserat_solver.consts as consts
import cosserat_solver.fourier as fourier
from cosserat_solver.integrator import Integrator


def generate_trace(
    x,
    material_params: dict,
    source,
    ft_params: dict,
    digits_precision=consts.COMPUTE_PRECISION,
    output_dir="OUTPUT_FILES",
    trace_prefix="AA.S0001.S2",
    save_to_file=False,
):
    """
    Generate a time-domain trace from the given parameters.

    Arguments:
    x: np.ndarray
        The spatial location where the trace is computed.
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
    source: Source
        The source object with a method `spectrum(omega)` that returns the source magnitude at frequency omega, and a method
            `direction()` that returns the source direction as a np.ndarray.
    ft_params: dict
        A dictionary containing the parameters for the Fourier transform. It should contain the following keys:
        - 'dt': The time step size for the output trace.
        - 'N': The number of time samples for the output trace.
        - 'oversample_rate': An integer specifying the oversampling rate for internal frequency sampling.
    digits_precision: int
        The number of digits of precision for intermediate mpmath calculations.
    output_dir: str
        The directory where the output files will be saved. Ignored if save_to_file is False. Default is "OUTPUT_FILES".
    trace_prefix: str
        The prefix for the trace file names. Default is "AA.S0001.S2". If multiple seismograms are generated, they should be notated as
            AA.S0001.S2, AA.S0002.S2, etc.
    save_to_file: bool
        Whether to save the generated trace to a file.

    Returns:
    times: np.ndarray
        The time samples corresponding to the trace.
    traces: dict
        A dictionary containing the trace components:
        - 'BXX': Trace component in the X direction.
        - 'BXZ': Trace component in the Z direction.
        - 'BYY': Trace component in the rotation direction.
    """

    # Unpack material parameters
    if not all(
        key in material_params
        for key in ["rho", "lam", "mu", "nu", "J", "lam_c", "mu_c", "nu_c"]
    ):
        msg = "Missing material parameters"
        raise ValueError(msg)
    rho = material_params["rho"]
    lam = material_params["lam"]
    mu = material_params["mu"]
    nu = material_params["nu"]
    J = material_params["J"]
    lam_c = material_params["lam_c"]
    mu_c = material_params["mu_c"]
    nu_c = material_params["nu_c"]

    # Create the integrator
    integrator = Integrator(
        rho, lam, mu, nu, J, lam_c, mu_c, nu_c, digits_precision=digits_precision
    )

    def frequency_domain_func(omega: float) -> np.ndarray:
        """
        The frequency domain function to be inverse transformed.

        Computed as G * source_spectrum, where source_spectrum is the magnitude of the source in frequency domain
        and G is the Green's function evaluated at the given frequency and location.
        """

        G_omega = integrator.greens_x_omega(x, omega)

        source_mag = source.spectrum(omega)
        return G_omega * source_mag

    greens_time = fourier.itransform(frequency_domain_func, ft_params)[
        ::-1, :, :
    ]  # shape (N, 3, 3)
    # need to reverse time axis due to conventions

    # Project onto source direction
    source_dir = source.direction()  # shape (3,)
    trace = np.einsum("tij,j->ti", greens_time, source_dir)  # shape (N, 3)

    traces = {
        "BXX": np.real(trace[:, 0]),
        "BXZ": np.real(trace[:, 1]),
        "BXY": np.real(trace[:, 2]),
    }

    times = np.arange(ft_params.get("N")) * ft_params.get("dt")

    if save_to_file:
        components = [("BXX", "semd"), ("BXZ", "semd"), ("BXY", "semr")]
        for channel, ext in components:
            if output_dir != ".":
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = f"{output_dir}/{trace_prefix}.{channel}.{ext}"
            else:
                filename = f"{trace_prefix}.{channel}.{ext}"

            with open(filename, "w") as f:
                for t, val in zip(times, traces[channel], strict=False):
                    f.write(f"{t:.6f} {val:.6e}\n")

    return times, traces
