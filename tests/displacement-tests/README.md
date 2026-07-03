A displacement test is specified using a folder in the `data` folder, with the folder name added to `SIMULATIONS` in `conftest.py`.
The folder must contain the following:
- A file `Snakefile` which runs the solver, so that running `snakemake -c1` produces all of the required seismogram traces in the folder `OUTPUT_FILES`.
- A subfolder `reference_traces` which contains the reference seismogram traces.
- A file `trace_list.csv` where each line is of the form `(reference trace filename), (generated trace filename)`, for instance, `DB.X55.S3.BXX.semr,AA.S0001.S2.BXX.semr`. The subfolders (`reference_traces`, `OUTPUT_FILES`, respectively) do not need to be specified.
- Any auxiliary files needed to run the Cosserat Solver simulations.
- Ideally, any auxiliary files needed to regenerate the reference traces, though this is not required.

Displacement tests will not run in pytest without the flag `--displacement-tests` due to the tendency for long simulations.
The maximum runtime for a displacement test is 5 minutes.
