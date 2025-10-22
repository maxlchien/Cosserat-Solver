from __future__ import annotations

import argparse

import cosserat_solver.read_yaml
import cosserat_solver.ricker
import cosserat_solver.trace_generator


def main() -> None:
    # set up argument parser
    parser = argparse.ArgumentParser(
        description="Cosserat Solver Command Line Interface"
    )
    # add optional --yaml <filename> argument
    parser.add_argument(
        "--yaml", type=str, required=False, help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    if args.yaml:
        (
            material_params,
            source_params,
            ft_params,
            digits_precision,
            seismogram_locations,
        ) = cosserat_solver.read_yaml.read(args.yaml)

    print(material_params)
    print(source_params)
    print(ft_params)
    print(digits_precision)
    print(seismogram_locations)

    if source_params.get("type") == "Ricker":
        source = cosserat_solver.ricker.Ricker(source_params)

    for i, location in enumerate(seismogram_locations):
        cosserat_solver.trace_generator.generate_trace(
            location,
            material_params,
            source,
            ft_params,
            digits_precision,
            trace_prefix=f"AA.S{str(i + 1).zfill(4)}.S2",
            save_to_file=True,
        )
