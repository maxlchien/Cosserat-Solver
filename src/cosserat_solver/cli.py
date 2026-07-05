from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime

from loguru import logger

import cosserat_solver.read_yaml
import cosserat_solver.ricker
import cosserat_solver.trace_generator
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
    # add optional --yaml <filename> argument
    parser.add_argument(
        "--yaml", type=str, required=False, help="Path to the YAML configuration file."
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

    logger.info("Starting Cosserat Solver CLI")
    logger.debug("Arguments: {args}", args=args)
    logger.info("Log file: {logfile}", logfile=logfile)

    # Validate mutually exclusive OpenMP flags
    if args.force_use_openmp and args.force_no_openmp:
        logger.error("--force-use-openmp and --force-no-openmp are mutually exclusive.")
        sys.exit(1)

    # Handle backend selection
    use_fortran = True
    if args.use_python_backend:
        # Explicitly requested Python backend
        use_fortran = False
        logger.info("Using Python backend (as requested)")
    elif not FORTRAN_AVAILABLE:
        # Fortran not available
        if args.allow_python_backend:
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
                "       Use --allow-python-backend to fallback to Python (slower)."
            )
            logger.info("       Use --use-python-backend to explicitly use Python.")
            sys.exit(1)
    else:
        # Fortran available and will be used
        logger.info("Using Fortran backend")

    if args.yaml:
        logger.info(f"Reading configuration from YAML file: {args.yaml}")
        (
            dim,
            material_params,
            source_params,
            ft_params,
            digits_precision,
            seismogram_locations,
        ) = cosserat_solver.read_yaml.read(args.yaml)

        logger.info("Dimension: {dim}", dim=dim)

        logger.info(
            "Material type: {material_type}",
            material_type=material_params.get("material_type", "N/A"),
        )

        logger.info("Material parameters:")
        logger.info("   Rho: {rho}", rho=material_params.get("rho", "N/A"))
        logger.info("   Lambda: {lam}", lam=material_params.get("lam", "N/A"))
        logger.info("   Mu: {mu}", mu=material_params.get("mu", "N/A"))
        logger.info("   Nu: {nu}", nu=material_params.get("nu", "N/A"))
        logger.info("   J: {J}", J=material_params.get("J", "N/A"))
        logger.info(
            "   Lambda (coupled): {lam_c}", lam_c=material_params.get("lam_c", "N/A")
        )
        logger.info("   Mu (coupled): {mu_c}", mu_c=material_params.get("mu_c", "N/A"))
        logger.info("   Nu (coupled): {nu_c}", nu_c=material_params.get("nu_c", "N/A"))

        logger.info("Source parameters:")
        logger.info(
            "   Type: {type}",
            type=source_params.get("type", "N/A"),
        )
        logger.info(
            "   Central Frequency f0 (Hz): {f0}",
            f0=source_params.get("f0", "N/A"),
        )
        logger.info(
            "   Factor: {factor}",
            factor=source_params.get("factor", "N/A"),
        )
        logger.info(
            "   Tshift: {tshift}",
            tshift=source_params.get("tshift", "N/A"),
        )
        logger.info("   f: {f}", f=source_params.get("f", "N/A"))
        logger.info("   fc: {fc}", fc=source_params.get("fc", "N/A"))
        logger.info("   Angle: {angle}", angle=source_params.get("angle", "N/A"))

        logger.info("Fourier parameters:")
        logger.info("   dt: {dt}", dt=ft_params.get("dt", "N/A"))
        logger.info("   N: {N}", N=ft_params.get("N", "N/A"))
        logger.info(
            "   Refinement factor: {extension_factor}",
            extension_factor=ft_params.get("extension_factor", "N/A"),
        )
        logger.info(
            "   Extension factor: {extension_factor}",
            extension_factor=ft_params.get("extension_factor", "N/A"),
        )

        logger.info(
            "Digits precision: {digits_precision} (only relevant for 2D Python backend)",
            digits_precision=digits_precision,
        )

        logger.info("Seismogram locations:")
        for location in seismogram_locations:
            logger.info("   {location}", location=location)
        logger.info("Finished reading configuration")

    if source_params.get("type") == "Ricker":
        if dim == 2:
            source = cosserat_solver.ricker.Ricker2D(source_params)
        elif dim == 3:
            source = cosserat_solver.ricker.Ricker3D(source_params)

    logger.info("=" * 40)
    logger.info("Beginning trace generation step at {time}", time=datetime.now())
    logger.info("=" * 40)
    trace_generation_start = time.perf_counter()

    for i, location in enumerate(seismogram_locations):
        logger.debug("=" * 40)
        logger.info(
            "Generating traces for seismogram {i} at location {location}",
            i=i + 1,
            location=location,
        )
        logger.debug("=" * 40)
        cosserat_solver.trace_generator.generate_trace(
            location,
            dim,
            material_params,
            source,
            ft_params,
            digits_precision,
            output_dir=args.o if args.o else "OUTPUT_FILES",
            trace_prefix=f"AA.S{str(i + 1).zfill(4)}.S2",
            save_to_file=True,
            use_fortran=use_fortran,
            force_use_openmp=args.force_use_openmp,
            force_no_openmp=args.force_no_openmp,
        )
    logger.info("=" * 40)
    logger.info("Finished trace generation step at {time}", time=datetime.now())
    logger.info(
        "Generated {n} traces in {duration:.2f} seconds",
        n=len(seismogram_locations),
        duration=time.perf_counter() - trace_generation_start,
    )
    logger.info("=" * 40)
