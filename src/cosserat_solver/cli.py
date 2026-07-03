from __future__ import annotations

import argparse
import sys
import warnings

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

    args = parser.parse_args()

    # Validate mutually exclusive OpenMP flags
    if args.force_use_openmp and args.force_no_openmp:
        print(
            "ERROR: --force-use-openmp and --force-no-openmp are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Handle backend selection
    use_fortran = True
    if args.use_python_backend:
        # Explicitly requested Python backend
        use_fortran = False
        print("Using Python backend (as requested)")
    elif not FORTRAN_AVAILABLE:
        # Fortran not available
        if args.allow_python_backend:
            use_fortran = False
            warnings.warn(
                "Fortran backend not available. Falling back to Python backend. "
                "This will be significantly slower (~21x).",
                UserWarning,
                stacklevel=2,
            )
            print("WARNING: Using Python backend (Fortran unavailable)")
        else:
            print("ERROR: Fortran backend is not available.", file=sys.stderr)
            print(
                "       The Fortran backend is required for performance.",
                file=sys.stderr,
            )
            print(
                "       Use --allow-python-backend to fallback to Python (slower).",
                file=sys.stderr,
            )
            print(
                "       Use --use-python-backend to explicitly use Python.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Fortran available and will be used
        print("Using Fortran backend")

    if args.yaml:
        (
            dim,
            material_params,
            source_params,
            ft_params,
            digits_precision,
            seismogram_locations,
        ) = cosserat_solver.read_yaml.read(args.yaml)

    print(dim)
    print(material_params)
    print(source_params)
    print(ft_params)
    print(digits_precision)
    print(seismogram_locations)

    if source_params.get("type") == "Ricker":
        if dim == 2:
            source = cosserat_solver.ricker.Ricker2D(source_params)
        elif dim == 3:
            source = cosserat_solver.ricker.Ricker3D(source_params)

    for i, location in enumerate(seismogram_locations):
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
