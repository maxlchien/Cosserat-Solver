from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import sys
import time
import warnings
from datetime import datetime

from loguru import logger

import cosserat_solver.read_yaml
import cosserat_solver.ricker
import cosserat_solver.trace_generator
from cosserat_solver import consts
from cosserat_solver.greens_wrapper import FORTRAN_AVAILABLE


def main() -> None:
    # set up argument parser
    parser = argparse.ArgumentParser(
        description="Cosserat Solver Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Backend Selection:
  By default, the Fortran backend is required for performance.
  Use --allow-python-backend to fallback to Python if Fortran is unavailable.
  Use --use-python-backend to explicitly use Python (slower, higher precision).
        """,
    )
    # --yaml <filename> argument. can be optional later if we add specfem_config reading
    parser.add_argument(
        "--yaml", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--o",
        type=str,
        required=False,
        help="Output directory for generated traces. Default is 'OUTPUT_FILES'.",
    )
    parser.add_argument(
        "--allow-python-backend",
        action="store_true",
        help="Allow fallback to Python backend if Fortran is unavailable (with warning).",
    )
    parser.add_argument(
        "--use-python-backend",
        action="store_true",
        help="Explicitly use Python backend (slower, arbitrary precision). Disables Fortran.",
    )
    parser.add_argument(
        "--force-use-openmp",
        action="store_true",
        help="Force OpenMP parallelization even for small arrays (incompatible with --force-no-openmp).",
    )
    parser.add_argument(
        "--force-no-openmp",
        action="store_true",
        help="Disable OpenMP parallelization even for large arrays (incompatible with --force-use-openmp).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info logging to stderr and debug logging to log file. Default is warning level to stderr and info level to log file.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to stderr and log file. Default is warning level to stderr and info level to log file.",
    )

    args = parser.parse_args()

    logfile = (
        args.o + "/cosserat_solver.log"
        if args.o
        else "OUTPUT_FILES/cosserat_solver.log"
    )
    # set up the logger
    logger.remove()
    if args.debug:
        logger.add(sys.stderr, format="{time} - {level} - {message}", level="DEBUG")
        logger.add(
            logfile, format="{time} - {level} - {message}", level="DEBUG", mode="w"
        )
    elif args.verbose:
        logger.add(sys.stderr, format="{time} - {level} - {message}", level="INFO")
        logger.add(
            logfile, format="{time} - {level} - {message}", level="DEBUG", mode="w"
        )
    else:
        logger.add(sys.stderr, format="{level} - {message}", level="WARNING")
        logger.add(logfile, format="{message}", level="INFO", mode="w")

    logger.info("Starting Cosserat Solver")
    logger.debug("Arguments: {args}", args=args)
    logger.info("Log file: {logfile}", logfile=logfile)

    logger.info(f"Reading configuration from YAML file: {args.yaml}")
    (
        dim,
        material_type,
        backend_from_yaml,
        material_params,
        sources,
        simulation_params,
        digits_precision,
        receivers,
        full_yaml,
    ) = cosserat_solver.read_yaml.read(args.yaml)
    logger.info("Configuration read successfully from YAML file.")
    logger.info("Parameters:")
    logger.info("=" * 40)

    if dim == consts.DIMENSION_2D:
        logger.info("Dimension: 2D")
    elif dim == consts.DIMENSION_3D:
        logger.info("Dimension: 3D")
    else:
        logger.error("Invalid dimension: {dim}", dim=dim)
        sys.exit(1)

    if material_type == consts.MATERIAL_TYPE_COSSERAT:
        logger.info("Material type: Cosserat")
    elif material_type == consts.MATERIAL_TYPE_ELASTIC:
        logger.info("Material type: Elastic")
    else:
        logger.error(
            "Invalid material type: {material_type}", material_type=material_type
        )
        sys.exit(1)

    backend = None
    if args.use_python_backend:
        backend_from_flag = consts.BACKEND_PYTHON
    elif args.allow_python_backend:
        backend_from_flag = consts.BACKEND_AUTO
    else:
        backend_from_flag = None  # No explicit backend flag provided
    if (
        "backend" in full_yaml
    ):  # if backend is specified by the file and as a flag, prefer the flag
        if backend_from_flag is not None and backend_from_flag != backend_from_yaml:
            logger.warning(
                "Backend specified in YAML ({backend_yaml}) and via command line flag ({backend_flag}) are different. Using command line flag.",
                backend_yaml=backend_from_yaml,
                backend_flag=backend_from_flag,
            )
            backend = backend_from_flag
        else:
            backend = backend_from_yaml
    else:
        if backend_from_flag is not None:
            backend = backend_from_flag
        else:
            backend = backend_from_yaml  # if no backend is specified in the yaml, this will be BACKEND_FORTRAN
    if backend == consts.BACKEND_FORTRAN:
        logger.info("Backend: Fortran")
    elif backend == consts.BACKEND_PYTHON:
        logger.info("Backend: Python")
    elif backend == consts.BACKEND_AUTO:
        logger.info("Backend: Auto (Fortran if available, otherwise Python)")
    else:
        logger.error("Invalid backend: {backend}", backend=backend)
        sys.exit(1)

    # we intentionally use .get() with no default value since all of these should be present
    logger.info("Material parameters:")
    logger.info("   Rho: {rho}", rho=material_params.get("rho"))
    logger.info("   Lambda: {lam}", lam=material_params.get("lam"))
    logger.info("   Mu: {mu}", mu=material_params.get("mu"))
    if material_type == consts.MATERIAL_TYPE_COSSERAT:
        logger.info("   Nu: {nu}", nu=material_params.get("nu"))
        logger.info("   J: {J}", J=material_params.get("J"))
        logger.info("   Lambda (coupled): {lam_c}", lam_c=material_params.get("lam_c"))
        logger.info("   Mu (coupled): {mu_c}", mu_c=material_params.get("mu_c"))
        logger.info("   Nu (coupled): {nu_c}", nu_c=material_params.get("nu_c"))

    logger.info("Sources:")
    for i, source in enumerate(sources):
        logger.info("   Source {n}:", n=i + 1)
        logger.info("      Location: {location}", location=source.get("location"))
        source_type = source.get("type")
        if source_type == consts.SOURCE_TYPE_RICKER:
            logger.info("      Type: Ricker")
        else:
            err = f"Invalid source type: {source_type}. Must be 'Ricker'."
            logger.error(err)
            raise ValueError(err)
        logger.info("      Type: {type}", type=source.get("type"))
        logger.info(
            "      Central Frequency f0 (Hz): {f0}",
            f0=source.get("f0"),
        )
        logger.info(
            "      Factor: {factor}",
            factor=source.get("factor"),
        )
        logger.info(
            "      Tshift: {tshift}",
            tshift=source.get("tshift"),
        )
        logger.info("      f: {f}", f=source.get("f"))
        logger.info("      fc: {fc}", fc=source.get("fc"))
        if dim == consts.DIMENSION_2D:
            logger.info("      Angle: {angle}", angle=source.get("angle"))

    logger.info(
        "Digits precision: {digits_precision} (only relevant for 2D Python backend)",
        digits_precision=digits_precision,
    )

    logger.info("Receivers:")
    logger.info("   Network: {network}", network=receivers.get("network"))
    for i, receiver in enumerate(receivers.get("receiver_list")):
        logger.info("   Receiver {i}: {name}", i=i + 1, name=receiver.get("name"))
        logger.info("      Location: {location}", location=receiver.get("location"))
        logger.info("      Seismograms to output:")
        for seismogram_type in receiver.get("seismogram_type"):
            if seismogram_type == consts.SEISMOGRAM_TYPE_DISPLACEMENT:
                logger.info("         - Displacement")
            elif seismogram_type == consts.SEISMOGRAM_TYPE_ROTATION:
                logger.info("         - Rotation")
            else:
                err = f"Invalid seismogram type: {seismogram_type}. Must be 0 (Displacement) or 1 (Rotation)."
                logger.error(err)
                raise ValueError(err)

    # Handle backend selection
    use_fortran = True
    if backend == consts.BACKEND_PYTHON:
        # Explicitly requested Python backend
        use_fortran = False
        logger.info("Using Python backend (as requested)")
    elif not FORTRAN_AVAILABLE:
        # Fortran not available
        if backend == consts.BACKEND_AUTO:
            use_fortran = False
            warnings.warn(
                "Fortran backend not available. Falling back to Python backend. "
                "This will be significantly slower for 2D cases (~21x).",
                UserWarning,
                stacklevel=2,
            )
            logger.warning("WARNING: Using Python backend (Fortran unavailable)")
        else:
            logger.error("ERROR: Fortran backend is not available.")
            logger.error("       The Fortran backend is required for performance.")
            logger.error(
                "       Use --allow-python-backend or set backend: auto in params.yaml to fallback to Python (slower)."
            )
            logger.info(
                "       Use --use-python-backend or set backend: python in params.yaml to explicitly use Python."
            )
            sys.exit(1)
    else:
        # Fortran available and will be used
        logger.info("Using Fortran backend")

    source_objects = []
    earliest_start_time = float("inf")
    for source in sources:
        if source.get("type") == consts.SOURCE_TYPE_RICKER:
            if dim == consts.DIMENSION_2D:
                obj = cosserat_solver.ricker.Ricker2D(source)
            elif dim == consts.DIMENSION_3D:
                obj = cosserat_solver.ricker.Ricker3D(source)
            else:
                err = f"Invalid dimension: {dim}. Must be 2 or 3."
                logger.error(err)
                raise ValueError(err)
        if obj.start_time < earliest_start_time:
            earliest_start_time = obj.start_time
        source_objects.append(obj)
    if len(source_objects) == 0:
        err = (
            "No valid sources found. At least one source of type 'Ricker' is required."
        )
        logger.error(err)
        raise ValueError(err)

    logger.info("General simulation parameters:")
    logger.info("   dt: {dt}", dt=simulation_params.get("dt"))
    logger.info("   N: {N}", N=simulation_params.get("N"))
    if simulation_params.get("t0") is None:
        logger.info(
            "   t0: None (will be set to the earliest source start time: {earliest_start_time})",
            earliest_start_time=earliest_start_time,
        )
        simulation_params["t0"] = earliest_start_time
    else:
        logger.info("   t0: {t0}", t0=simulation_params.get("t0"))
    logger.info(
        "   Refinement factor: {refinement_factor}",
        refinement_factor=simulation_params.get("refinement_factor"),
    )
    logger.info(
        "   Extension factor: {extension_factor}",
        extension_factor=simulation_params.get("extension_factor"),
    )

    logger.info("Getting OpenMP information")
    # Validate mutually exclusive OpenMP flags
    if args.force_use_openmp and args.force_no_openmp:
        logger.error("--force-use-openmp and --force-no-openmp are mutually exclusive.")
        sys.exit(1)
    info = {}
    for libname in ("libgomp.so.1", "libomp.so", "libiomp5.so", "libomp.dylib"):
        try:
            lib = ctypes.CDLL(libname)
            info["available"] = True
            info["max_threads"] = lib.omp_get_max_threads()
            info["num_procs"] = lib.omp_get_num_procs()
            info["runtime_lib"] = libname
            break
        except OSError:
            continue
    if info:
        logger.info(
            "OpenMP runtime library found: {runtime_lib}",
            runtime_lib=info["runtime_lib"],
        )
        logger.info(
            "OpenMP max threads: {max_threads}", max_threads=info["max_threads"]
        )
        logger.info(
            "OpenMP number of processors: {num_procs}", num_procs=info["num_procs"]
        )
    else:
        logger.info(
            "No OpenMP runtime library found at libgomp.so.1, libomp.so, libiomp5.so, or libomp.dylib."
        )

    logger.info("=" * 40)
    logger.info("Beginning trace generation step at {time}", time=datetime.now())
    logger.info("=" * 40)
    trace_generation_start = time.perf_counter()

    for i, receiver in enumerate(receivers.get("receiver_list")):
        logger.debug("=" * 40)
        logger.info(
            "Generating traces for seismogram {name} at location {location} ({n} of {total})",
            name=receiver.get("name"),
            location=receiver.get("location"),
            n=i + 1,
            total=len(receivers.get("receiver_list")),
        )
        logger.debug("=" * 40)
        cosserat_solver.trace_generator.generate_trace(
            receiver,
            dim,
            material_type,
            material_params,
            source_objects,
            simulation_params,
            digits_precision,
            output_dir=args.o if args.o else "OUTPUT_FILES",
            trace_prefix=f"{receivers.get('network')}.{receiver.get('name')}.S{2 if dim == consts.DIMENSION_2D else 3}",
            save_to_file=True,
            use_fortran=use_fortran,
            force_use_openmp=args.force_use_openmp,
            force_no_openmp=args.force_no_openmp,
        )
    logger.info("=" * 40)
    logger.info("Finished trace generation step at {time}", time=datetime.now())
    logger.info(
        "Generated {n} traces in {duration:.2f} seconds",
        n=len(receivers.get("receiver_list")),
        duration=time.perf_counter() - trace_generation_start,
    )
    logger.info("=" * 40)
