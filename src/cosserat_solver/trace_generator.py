from __future__ import annotations

import os

import numpy as np

import cosserat_solver.consts as consts
import cosserat_solver.fourier as fourier
from cosserat_solver.greens_wrapper import get_greens_callback
from cosserat_solver.source import SourceSpectrum


def generate_trace(
    x,
    material_params: dict,
    source: SourceSpectrum,
    ft_params: dict,
    digits_precision=consts.COMPUTE_PRECISION,
    output_dir="OUTPUT_FILES",
    trace_prefix="AA.S0001.S2",
    save_to_file=False,
    use_fortran=True,
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
    source: SourceSpectrum
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
    use_fortran: bool
        Whether to use Fortran backend (faster) or Python backend (slower, higher precision). Default is True.

    Returns:
    times: np.ndarray
        The time samples corresponding to the trace.
    traces: dict
        A dictionary containing the trace components:
        - 'BXX': Trace component in the X direction.
        - 'BXZ': Trace component in the Z direction.
        - 'BYY': Trace component in the rotation direction.
    """

    # for safety since we may edit this
    ft_params = ft_params.copy()

    frequency_domain_func = get_greens_callback(
        x,
        material_params,
        source,
        use_fortran=use_fortran,
        digits_precision=digits_precision,
    )

    if "t0" not in ft_params:
        ft_params["t0"] = source.t0()

    times, greens_time = fourier.cont_ifft(frequency_domain_func, ft_params)

    # Project onto source direction
    source_dir = source.direction()  # shape (3,)
    trace = np.einsum("tij,j->ti", greens_time, source_dir)  # shape (N, 3)

    traces = {
        "BXX": np.real(trace[:, 0]),
        "BXZ": np.real(trace[:, 1]),
        "BXY": np.real(trace[:, 2]),
    }

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
