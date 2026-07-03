from __future__ import annotations

import os

import numpy as np

import cosserat_solver.consts as consts
import cosserat_solver.fourier as fourier
from cosserat_solver.greens_wrapper import get_greens_callback, get_material_tag
from cosserat_solver.source import SourceSpectrum

channels_2d = ["BXX.semd", "BXZ.semd", "BXY.semr"]
channels_3d = ["BXX.semd", "BXY.semd", "BXZ.semd", "BXX.semr", "BXY.semr", "BXZ.semr"]


def generate_trace(
    x,
    dim: int,
    material_params: dict,
    source: SourceSpectrum,
    ft_params: dict,
    digits_precision=consts.COMPUTE_PRECISION,
    output_dir="OUTPUT_FILES",
    trace_prefix="AA.S0001.S2",
    save_to_file=False,
    use_fortran=True,
    force_use_openmp: bool = False,
    force_no_openmp: bool = False,
):
    """
    Generate a time-domain trace from the given parameters.

    Arguments:
    x: np.ndarray
        The spatial location where the trace is computed.
    dim: int
        The dimension of the problem. Either 2 or 3.
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
        - 'extension_factor': An integer specifying the factor for extending the window.
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
    force_use_openmp: bool
        If True, force OpenMP parallelization even for small arrays.
    force_no_openmp: bool
        If True, disable OpenMP parallelization even for large arrays.

    Returns:
    times: np.ndarray
        The time samples corresponding to the trace.
    traces: dict
        A dictionary containing the trace components:
        In 2D:
        - 'BXX.semd': Trace component in the X direction.
        - 'BXZ.semd': Trace component in the Z direction.
        - 'BXY.semr': Trace component in the rotation direction.

        In 3D:
        - 'BXX.semd': Trace component in the X direction.
        - 'BXY.semd': Trace component in the Y direction.
        - 'BXZ.semd': Trace component in the Z direction.
        - 'BXX.semr': Trace component in the X rotation direction.
        - 'BXY.semr': Trace component in the Y rotation direction.
        - 'BXZ.semr': Trace component in the Z rotation direction.
    """

    if dim not in (2, 3):
        err = f"Invalid dimension {dim}. Dimension must be either 2 or 3."
        raise ValueError(err)
    if len(x) != dim:
        err = f"Spatial location x must have length {dim} for dimension {dim}."
        raise ValueError(err)

    if "material_type" not in material_params:
        err = "Material type must be specified in material_params with key 'material_type'."
        raise ValueError(err)
    material_tag = get_material_tag(material_params["material_type"])

    # for safety since we may edit this
    ft_params = ft_params.copy()

    frequency_domain_func = get_greens_callback(
        x,
        dim,
        material_params,
        source,
        use_fortran,
        material_tag,
        digits_precision=digits_precision,
        force_use_openmp=force_use_openmp,
        force_no_openmp=force_no_openmp,
    )

    if "t0" not in ft_params:
        ft_params["t0"] = source.t0()

    times, greens_time = fourier.cont_ifft(frequency_domain_func, ft_params)

    # Project onto source direction
    source_dir = source.direction()  # shape (3,) or (6,)
    trace = np.einsum("tij,j->ti", greens_time, source_dir)  # shape (N, 3) or (N, 6)

    channels = channels_2d if dim == 2 else channels_3d

    traces = {channel: np.real(trace[:, i]) for i, channel in enumerate(channels)}

    if save_to_file:
        for channel in channels:
            if output_dir != ".":
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = f"{output_dir}/{trace_prefix}.{channel}"
            else:
                filename = f"{trace_prefix}.{channel}"

            with open(filename, "w") as f:
                for t, val in zip(times, traces[channel], strict=False):
                    f.write(f"{t:.6f} {val:.6e}\n")

    return times, traces
