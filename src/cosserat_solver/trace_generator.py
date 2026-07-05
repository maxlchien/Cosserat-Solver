from __future__ import annotations

import os

import numpy as np
from loguru import logger

import cosserat_solver.consts as consts
import cosserat_solver.fourier as fourier
from cosserat_solver.greens_wrapper import get_greens_callback
from cosserat_solver.source import SourceSpectrum

channels_2d = ["XX.semd", "XZ.semd", "XY.semr"]
channels_3d = ["XX.semd", "XY.semd", "XZ.semd", "XX.semr", "XY.semr", "XZ.semr"]


def get_prefix(dt: float) -> str:
    """
    Get the prefix for the trace file names based on the time step size.

    Parameters:
        dt (float): The time step size.

    Returns:
        SEED-compliant seismogram channel code prefix.
    """
    if dt == 0:
        msg = "Time step size dt cannot be zero."
        logger.error(msg)
        raise ValueError(msg)
    hz = 1 / dt
    if hz < 0.01:
        return "U"
    if hz < 0.1:
        return "V"
    if hz < 1:
        return "L"
    if hz < 80:
        return "B"
    return "H"


def generate_trace(
    receiver: dict,
    dim: int,
    material_type: int,
    material_params: dict,
    sources: list[SourceSpectrum],
    simulation_params: dict,
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
    receiver: dict
        A dictionary containing the receiver parameters.
    dim: int
        The dimension of the problem. Either 2 or 3.
    material_type: int
        The type of material. Either 0 (elastic) or 1 (Cosserat).
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
    sources: list[SourceSpectrum]
        A list of source objects, each with a method `spectrum(omega)` that returns the source magnitude at frequency omega, and a method
            `direction()` that returns the source direction as a np.ndarray.
    simulation_params: dict
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
        A dictionary containing the trace components specified.
        At most, they are as follows ('B' may be replaced by another SEED code depending on dt):
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
        logger.error(err)
        raise ValueError(err)
    if len(receiver.get("location")) != dim:
        err = f"Spatial location x must have length {dim} for dimension {dim}, but got length {len(receiver.get('location'))}."
        logger.error(err)
        raise ValueError(err)

    channels_without_prefix = channels_2d if dim == 2 else channels_3d
    prefix = get_prefix(simulation_params.get("dt"))

    channels = [f"{prefix}{channel}" for channel in channels_without_prefix]

    traces = {}
    for channel in channels:
        traces[channel] = np.zeros(int(simulation_params.get("N")), dtype=np.float64)

    for source in sources:
        if not isinstance(source, SourceSpectrum):
            err = f"Source {source} is not an instance of SourceSpectrum."
            logger.error(err)
            raise TypeError(err)

        # for safety since we may edit this
        simulation_params_editable = simulation_params.copy()
        if "t0" not in simulation_params_editable:
            logger.debug("t0 not specified in simulation_params, using source.t0()")
            simulation_params_editable["t0"] = source.t0()

        frequency_domain_func = get_greens_callback(
            receiver.get("location") - source.location(),
            dim,
            material_params,
            source,
            use_fortran,
            material_type,
            digits_precision=digits_precision,
            force_use_openmp=force_use_openmp,
            force_no_openmp=force_no_openmp,
        )

        times, greens_time = fourier.cont_ifft(
            frequency_domain_func, simulation_params_editable
        )

        # Project onto source direction
        source_dir = source.direction()  # shape (3,) or (6,)
        trace = np.einsum(
            "tij,j->ti", greens_time, source_dir
        )  # shape (N, 3) or (N, 6)

        for i, channel in enumerate(channels):
            traces[channel] += np.real(
                trace[:, i]
            )  # accumulate contributions from all sources

    if save_to_file:
        if not os.path.exists(output_dir):
            logger.debug(
                "Output directory {output_dir} does not exist. Creating it.",
                output_dir=output_dir,
            )
            os.makedirs(output_dir)
        extensions_to_save = []
        if consts.SEISMOGRAM_TYPE_DISPLACEMENT in receiver.get("seismogram_type"):
            extensions_to_save.append(".semd")
        if consts.SEISMOGRAM_TYPE_ROTATION in receiver.get("seismogram_type"):
            extensions_to_save.append(".semr")
        for channel in channels:
            if not any(channel.endswith(ext) for ext in extensions_to_save):
                logger.debug(
                    "Skipping channel {channel} as it does not match the requested seismogram types.",
                    channel=channel,
                )
                continue
            if output_dir != ".":
                filename = f"{output_dir}/{trace_prefix}.{channel}"
                logger.debug("Saving trace to file: {filename}", filename=filename)
            else:
                filename = f"{trace_prefix}.{channel}"
                logger.debug("Saving trace to file: {filename}", filename=filename)

            with open(filename, "w") as f:
                for t, val in zip(times, traces[channel], strict=False):
                    f.write(f"{t:.6f} {val:.6e}\n")
            logger.debug(
                "Trace {channel} saved to file: {filename}",
                channel=channel,
                filename=filename,
            )

    return times, traces
