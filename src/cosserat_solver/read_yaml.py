from __future__ import annotations

import numpy as np
import yaml
from loguru import logger

import cosserat_solver.consts as consts


def _require_float(value: object, name: str) -> float:
    """Coerce a YAML scalar to float, raising ValueError with a clear message."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        msg = f"'{name}' in 'spherical_wavefield' must be a number, got {value!r}."
        logger.error(msg)
        raise ValueError(msg) from None


def _require_int(value: object, name: str, minimum: int) -> int:
    """Coerce an integer-valued YAML scalar, rejecting bools and non-integers."""
    if isinstance(value, bool):
        msg = f"'{name}' in 'spherical_wavefield' must be an integer, not a bool."
        logger.error(msg)
        raise ValueError(msg)
    try:
        is_integer_valued = float(value).is_integer()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        is_integer_valued = False
    if not is_integer_valued:
        msg = f"'{name}' in 'spherical_wavefield' must be an integer, got {value!r}."
        logger.error(msg)
        raise ValueError(msg)
    result = int(value)  # type: ignore[arg-type]
    if result < minimum:
        msg = f"'{name}' in 'spherical_wavefield' must be >= {minimum}, got {result}."
        logger.error(msg)
        raise ValueError(msg)
    return result


def _parse_spherical_wavefield(block: object, sources: list) -> dict:
    """
    Validate and normalize the optional 'spherical_wavefield' block.

    Returns a dictionary with every key populated (defaults applied). See the
    'spherical_wavefield' entry of :func:`read`'s docstring for the schema. All
    sources must sit exactly at the wavefield center.
    """
    if not isinstance(block, dict):
        msg = f"'spherical_wavefield' must be a mapping, got {type(block).__name__}."
        logger.error(msg)
        raise ValueError(msg)
    block = dict(block)  # copy so we can pop and detect leftover (extra) keys
    result: dict = {}

    # center (optional, default origin)
    if "center" in block:
        center = block.pop("center")
        if not isinstance(center, (list, np.ndarray)) or len(center) != 3:
            msg = f"'center' in 'spherical_wavefield' must be a length-3 list, got {center!r}."
            logger.error(msg)
            raise ValueError(msg)
        result["center"] = np.array([float(v) for v in center], dtype=float)
    else:
        result["center"] = np.zeros(3, dtype=float)

    # r_max (required, > 0)
    if "r_max" not in block:
        msg = "Missing 'r_max' key in 'spherical_wavefield'."
        logger.error(msg)
        raise KeyError(msg)
    r_max = _require_float(block.pop("r_max"), "r_max")
    if r_max <= 0.0:
        msg = f"'r_max' in 'spherical_wavefield' must be > 0, got {r_max}."
        logger.error(msg)
        raise ValueError(msg)
    result["r_max"] = r_max

    # num_radii (required, >= 1)
    if "num_radii" not in block:
        msg = "Missing 'num_radii' key in 'spherical_wavefield'."
        logger.error(msg)
        raise KeyError(msg)
    result["num_radii"] = _require_int(block.pop("num_radii"), "num_radii", 1)

    # r_min (optional; default r_max / num_radii so radii are dr, 2*dr, ..., r_max)
    if "r_min" in block:
        r_min = _require_float(block.pop("r_min"), "r_min")
        if r_min <= 0.0:
            msg = f"'r_min' in 'spherical_wavefield' must be > 0, got {r_min}."
            logger.error(msg)
            raise ValueError(msg)
        if r_min >= r_max:
            msg = f"'r_min' ({r_min}) must be < 'r_max' ({r_max}) in 'spherical_wavefield'."
            logger.error(msg)
            raise ValueError(msg)
        result["r_min"] = r_min
    else:
        result["r_min"] = r_max / result["num_radii"]
        logger.info(
            "Missing 'r_min' key in 'spherical_wavefield'. Defaulting to r_max / num_radii = {r_min}.",
            r_min=result["r_min"],
        )

    # num_theta (required, >= 2: theta spans [0, pi], a single point degenerates)
    if "num_theta" not in block:
        msg = "Missing 'num_theta' key in 'spherical_wavefield'."
        logger.error(msg)
        raise KeyError(msg)
    result["num_theta"] = _require_int(block.pop("num_theta"), "num_theta", 2)

    # num_phi (required, >= 1)
    if "num_phi" not in block:
        msg = "Missing 'num_phi' key in 'spherical_wavefield'."
        logger.error(msg)
        raise KeyError(msg)
    result["num_phi"] = _require_int(block.pop("num_phi"), "num_phi", 1)

    # steps_per_snapshot (required, >= 1)
    if "steps_per_snapshot" not in block:
        msg = (
            "Missing 'steps_per_snapshot' key in 'spherical_wavefield'. This is "
            "required; set it to 1 to write every output time sample."
        )
        logger.error(msg)
        raise KeyError(msg)
    result["steps_per_snapshot"] = _require_int(
        block.pop("steps_per_snapshot"), "steps_per_snapshot", 1
    )

    # directory (optional; None means the CLI resolves a default under --o)
    if "directory" in block:
        directory = block.pop("directory")
        if not isinstance(directory, str) or not directory:
            msg = f"'directory' in 'spherical_wavefield' must be a non-empty string, got {directory!r}."
            logger.error(msg)
            raise ValueError(msg)
        result["directory"] = directory
    else:
        result["directory"] = None

    # compression (optional, default gzip; null or "none" disables it)
    if "compression" in block:
        compression = block.pop("compression")
        if compression is None or (
            isinstance(compression, str) and compression.lower() == "none"
        ):
            result["compression"] = None
        elif isinstance(compression, str):
            result["compression"] = compression
        else:
            msg = f"'compression' in 'spherical_wavefield' must be a string or null, got {compression!r}."
            logger.error(msg)
            raise ValueError(msg)
    else:
        result["compression"] = "gzip"

    # write_xdmf (optional, default True)
    if "write_xdmf" in block:
        write_xdmf = block.pop("write_xdmf")
        if not isinstance(write_xdmf, bool):
            msg = f"'write_xdmf' in 'spherical_wavefield' must be a boolean, got {write_xdmf!r}."
            logger.error(msg)
            raise ValueError(msg)
        result["write_xdmf"] = write_xdmf
    else:
        result["write_xdmf"] = True

    # radial_chunk_size (optional, default None)
    if block.get("radial_chunk_size") is not None:
        result["radial_chunk_size"] = _require_int(
            block.pop("radial_chunk_size"), "radial_chunk_size", 1
        )
    else:
        block.pop("radial_chunk_size", None)
        result["radial_chunk_size"] = None

    # All sources must sit exactly at the wavefield center.
    for i, source in enumerate(sources):
        location = np.asarray(source["location"], dtype=float)
        if not np.allclose(location, result["center"]):
            msg = (
                f"'spherical_wavefield' requires all sources at the center "
                f"{result['center']}, but source {i + 1} is at {location}."
            )
            logger.error(msg)
            raise ValueError(msg)

    if block:
        logger.warning(
            f"Extra keys in 'spherical_wavefield' that are not recognized: {', '.join(block.keys())}. They will be ignored."
        )

    return result


def read(
    file_path: str,
) -> tuple[int, int, int, dict, list, dict, int, dict, dict | None, dict]:
    """
    Read a YAML file and return its contents as a dictionary.
    All substitution of default values should occur here.
    Some validation of parameters occurs.

    Parameters:
    file_path: str
        The path to the YAML file.

    Returns:

    dim: int
        The dimension of the problem (2 or 3).

    material_type: int
        The type of material (0 for elastic, 1 for Cosserat). Defaults to Cosserat.

    backend: int
        The backend to use (0 for auto, 1 for Fortran, 2 for Python). Defaults to Fortran.

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

    sources: list
        A list of dictionaries containing the source parameters:
        - 'type': Type of the source (e.g., "Ricker").
        - 'location': Location of the source as a list of floats.
        - 'f0': Dominant frequency of the Ricker wavelet in Hz.
        - 'f': Displacement ratio in 2D, or direction vector in 3D.
        - 'fc': Rotation ratio in 2D, or moment vector in 3D.
        - 'angle': Source angle in degrees (2D only).
        - 'factor': Amplitude scaling factor.
        - 'tshift': Time shift for the source in seconds.

    simulation_params: dict
        A dictionary containing the parameters for the Fourier transform. It should contain the following keys:
        - 'dt': The time step size for the output trace.
        - 'N': The number of time samples for the output trace.
        - 'extension_factor': An integer specifying the factor for extending the window.
        - 'refinement_factor': An integer specifying the factor for refining the window.
        - 't0': The start time for the output trace.

    digits_precision: int
        The number of digits of precision for intermediate mpmath calculations.

    receivers: dict
        A dictionary containing the receiver parameters:
        - `network`: The name of the network. Defaults to "AA".
        - `receiver_list`: A list of dictionaries, each containing:
            - `location`: A tuple of floats specifying the receiver's location.
            - `name`: An optional string specifying the receiver's name. If not provided, a default name will be generated.
            - 'seismogram_type`: An optional list specifying the type of seismogram to record. Defaults to both displacement and rotation for Cosserat materials, and only displacement for elastic materials.

    spherical_wavefield: dict | None
        The spherical wavefield output parameters, or None if the optional
        'spherical_wavefield' block is absent (no wavefield is written). When present,
        a normalized dictionary containing:
        - 'center': A length-3 numpy array; the wavefield center. All sources must be
          located here. Defaults to the origin.
        - 'r_min': The innermost radius (float > 0). Defaults to
          'r_max' / 'num_radii' when absent (so the radii become dr, 2*dr, ..., r_max).
        - 'r_max': The outermost radius (float > 0).
        - 'num_radii': The number of radial shells (int >= 1).
        - 'num_theta': The number of polar-angle samples in [0, pi] (int >= 2).
        - 'num_phi': The number of azimuth samples in [0, 2*pi) (int >= 1).
        - 'steps_per_snapshot': The stride over the N-sample output grid between
          written snapshots (int >= 1). Required; there is no default.
        - 'directory': The output directory (str), or None to default to a
          'spherical_wavefield' subdirectory of the CLI output directory.
        - 'compression': The HDF5 compression filter (str), or None to disable.
          Defaults to 'gzip'.
        - 'write_xdmf': Whether to write the ParaView XDMF sidecar (bool). Defaults
          to True.
        - 'radial_chunk_size': The radial chunk size bounding transient memory
          (int >= 1), or None for no chunking. Defaults to None.

    full_yaml: dict
        The full contents of the YAML file as a dictionary.
    """
    with open(file_path) as file:
        data = yaml.safe_load(file)
        data_copy = data.copy()  # make a copy of the original data to return later

    logger.info("YAML file read successfully from {file_path}.", file_path=file_path)

    logger.debug("YAML file contents: {data}", data=data)

    logger.info("Validating YAML file contents...")

    logger.debug("Reading dimension key...")
    try:
        dimension = data.pop("dimension")
    except KeyError as e:
        err = "Missing 'dimension' key in YAML file."
        logger.error(err)
        raise KeyError(err) from e
    if dimension == 2:
        dimension_code = consts.DIMENSION_2D
    elif dimension == 3:
        dimension_code = consts.DIMENSION_3D
    else:
        err = f"Invalid dimension {dimension}. Dimension must be specified as either 2 or 3."
        logger.error(err)
        raise ValueError(err)
    logger.debug(
        "Dimension {dimension} read successfully, with a code of {dimension_code}.",
        dimension=dimension,
        dimension_code=dimension_code,
    )

    logger.debug("Reading material_type key...")
    try:
        material_type = data.pop("material_type")
    except KeyError:
        logger.info(
            "Missing 'material_type' key in YAML file. Defaulting to 'cosserat'."
        )
        material_type = "cosserat"
    if material_type == "elastic":
        material_type_code = consts.MATERIAL_TYPE_ELASTIC
    elif material_type == "cosserat":
        material_type_code = consts.MATERIAL_TYPE_COSSERAT
    else:
        err = f"Invalid material_type {material_type}. Must be either 'elastic' or 'cosserat'."
        logger.error(err)
        raise ValueError(err)
    logger.debug(
        "Material type {material_type} read successfully, with a code of {material_type_code}.",
        material_type=material_type,
        material_type_code=material_type_code,
    )

    logger.debug("Reading backend key...")
    try:
        backend = data.pop("backend")
    except KeyError:
        logger.info("Missing 'backend' key in YAML file. Defaulting to 'fortran'.")
        backend = "fortran"
    if backend == "fortran":
        backend_code = consts.BACKEND_FORTRAN
    elif backend == "python":
        backend_code = consts.BACKEND_PYTHON
    elif backend == "auto":
        backend_code = consts.BACKEND_AUTO
    else:
        err = (
            f"Invalid backend {backend}. Must be either 'fortran', 'python', or 'auto'."
        )
        logger.error(err)
        raise ValueError(err)
    logger.debug(
        "Backend {backend} read successfully, with a code of {backend_code}.",
        backend=backend,
        backend_code=backend_code,
    )

    # from here, we assume that the dimension, material_type, and backend are valid

    logger.debug("Reading material_params key...")
    try:
        material_params_read = data.pop("material_params")
    except KeyError as e:
        err = "Missing 'material_params' key in YAML file."
        logger.error(err)
        raise KeyError(err) from e
    material_params = {}
    required_keys = set()
    substituted_keys = {}
    if material_type_code == consts.MATERIAL_TYPE_COSSERAT:
        required_keys = {"rho", "lam", "mu", "nu", "J", "lam_c", "mu_c", "nu_c"}
        substituted_keys = {}
    elif material_type_code == consts.MATERIAL_TYPE_ELASTIC:
        required_keys = {"rho", "lam", "mu"}
        substituted_keys = {"nu": 0.0, "J": 1.0, "lam_c": 0.0, "mu_c": 0.0, "nu_c": 0.0}
    logger.debug(
        "Required material parameters for material type {material_type}: {required_keys}",
        material_type=material_type,
        required_keys=required_keys,
    )
    logger.debug(
        "Substituted material parameters for material type {material_type}: {substituted_keys}",
        material_type=material_type,
        substituted_keys=substituted_keys,
    )
    if not required_keys.issubset(material_params_read.keys()):
        missing_keys = required_keys - material_params_read.keys()
        err = f"Missing required material parameters: {', '.join(missing_keys)}"
        logger.error(err)
        raise KeyError(err)
    for key in required_keys:
        try:
            param_value = material_params_read.pop(key)
            material_params[key] = float(param_value)
        except KeyError as e:
            err = f"Missing required material parameter '{key}' in 'material_params'."
            logger.error(err)
            raise KeyError(err) from e
        except ValueError as e:
            err = f"Invalid value for material parameter '{key}', got {param_value}. Must be a float."
            logger.error(err)
            raise ValueError(err) from e
    for key, default_value in substituted_keys.items():
        if key in material_params_read:
            logger.info(
                f"Material parameter '{key}' was provided but is unused for material type '{material_type}'. It will be ignored."
            )
            material_params_read.pop(key)
        material_params[key] = default_value
    if material_params_read:
        logger.warning(
            f"Extra keys in 'material_params' that are not required or recognized: {', '.join(material_params_read.keys())}. They will be ignored."
        )
    logger.debug(
        "Material parameters read successfully: {material_params}",
        material_params=material_params,
    )

    logger.debug("Reading sources key...")
    try:
        sources_read = data.pop("sources")
    except KeyError as e:
        err = "Missing 'sources' key in YAML file."
        logger.error(err)
        raise KeyError(err) from e
    sources = []
    for i, source_read in enumerate(sources_read):
        logger.debug("Reading source entry {i}...", i=i + 1)
        source = {}
        defaulted_source_keys = {}
        substituted_source_keys = {}
        try:
            source_type = source_read.pop("type")
        except KeyError as e:
            err = f"Missing 'type' key in source entry {i + 1}."
            logger.error(err)
            raise KeyError(err) from e
        if source_type == "Ricker":
            source["type"] = consts.SOURCE_TYPE_RICKER
            if dimension_code == consts.DIMENSION_2D:
                defaulted_source_keys = {
                    "location": np.array([0.0, 0.0], dtype=float),
                    "f0": 25.0,
                    "f": 1.0,
                    "fc": 1.0,
                    "factor": 1.0,
                    "angle": 0.0,
                    "tshift": 0.0,
                }
                substituted_source_keys = {}
            elif dimension_code == consts.DIMENSION_3D:
                defaulted_source_keys = {
                    "location": np.array([0.0, 0.0, 0.0], dtype=float),
                    "f0": 25.0,
                    "f": np.array([1.0, 1.0, 1.0], dtype=float),
                    "fc": np.array([1.0, 1.0, 1.0], dtype=float),
                    "factor": 1.0,
                    "tshift": 0.0,
                }
                substituted_source_keys = {"angle": 0.0}
        else:
            err = f"Invalid source type {source_type} in source entry {i + 1}. Must be 'Ricker'."
            logger.error(err)
            raise ValueError(err)
        logger.debug(
            "Defaulted source keys for source entry {i}: {defaulted_source_keys}",
            i=i + 1,
            defaulted_source_keys=defaulted_source_keys,
        )
        logger.debug(
            "Substituted source keys for source entry {i}: {substituted_source_keys}",
            i=i + 1,
            substituted_source_keys=substituted_source_keys,
        )
        for key, default_value in defaulted_source_keys.items():
            try:
                param_value = source_read.pop(key)
                if isinstance(default_value, np.ndarray):
                    if not isinstance(param_value, (list, np.ndarray)):
                        err = f"Invalid type for source parameter '{key}' in source entry {i + 1}, got {param_value}. Must be a list or numpy array."
                        logger.error(err)
                        raise ValueError(err)
                    if len(param_value) != len(default_value):
                        err = f"Invalid type for source parameter '{key}' in source entry {i + 1}, got {param_value}. Must be a list of length {len(default_value)}."
                        logger.error(err)
                        raise ValueError(err)
                    param_value = np.array(param_value, dtype=float)
                else:
                    param_value = float(param_value)
                source[key] = param_value
            except KeyError:
                logger.info(
                    f"Missing '{key}' key in source entry {i + 1}. Defaulting to {default_value}."
                )
                source[key] = default_value
        for key, default_value in substituted_source_keys.items():
            if key in source_read:
                logger.info(
                    f"Source parameter '{key}' was provided but is unused for dimension {dimension_code}. It will be ignored."
                )
                source_read.pop(key)
            source[key] = default_value
        if source_read:
            logger.warning(
                f"Extra keys in source entry {i + 1} that are not required or recognized: {', '.join(source_read.keys())}. They will be ignored."
            )
        if len(source["location"]) != dimension:
            err = f"Source location {source['location']} has length {len(source['location'])}, but dimension is {dimension}."
            logger.error(err)
            raise ValueError(err)
        sources.append(source)
    logger.debug("Sources read successfully: {sources}", sources=sources)

    logger.debug("Reading simulation_params key...")
    try:
        simulation_params_read = data.pop("simulation_params")
    except KeyError as e:
        err = "Missing 'simulation_params' key in YAML file."
        logger.error(err)
        raise KeyError(err) from e
    simulation_params = {}
    # required keys
    try:
        dt = simulation_params_read.pop("dt")
        simulation_params["dt"] = float(dt)
    except KeyError as e:
        err = "Missing 'dt' key in 'simulation_params'."
        logger.error(err)
        raise KeyError(err) from e
    except ValueError as e:
        err = (
            f"Invalid value for 'dt' in 'simulation_params', got {dt}. Must be a float."
        )
        logger.error(err)
        raise ValueError(err) from e
    try:
        N = simulation_params_read.pop("N")
        simulation_params["N"] = int(N)
    except KeyError as e:
        err = "Missing 'N' key in 'simulation_params'."
        logger.error(err)
        raise KeyError(err) from e
    except ValueError as e:
        err = f"Invalid value for 'N' in 'simulation_params', got {N}. Must be an integer."
        logger.error(err)
        raise ValueError(err) from e
    # optional keys
    try:
        t0 = simulation_params_read.pop("t0")
        simulation_params["t0"] = float(t0)
    except KeyError:
        logger.debug(
            "Missing 't0' key in 'simulation_params'. Defaulting to None, which will be set to the earliest source start time later."
        )
        simulation_params["t0"] = None
    except ValueError as e:
        err = (
            f"Invalid value for 't0' in 'simulation_params', got {t0}. Must be a float."
        )
        logger.error(err)
        raise ValueError(err) from e
    try:
        extension_factor = simulation_params_read.pop("extension_factor")
        simulation_params["extension_factor"] = int(extension_factor)
    except KeyError:
        logger.debug(
            "Missing 'extension_factor' key in 'simulation_params'. Defaulting to 1."
        )
        simulation_params["extension_factor"] = 1
    except ValueError as e:
        err = f"Invalid value for 'extension_factor' in 'simulation_params', got {extension_factor}. Must be an integer."
        logger.error(err)
        raise ValueError(err) from e
    try:
        refinement_factor = simulation_params_read.pop("refinement_factor")
        simulation_params["refinement_factor"] = int(refinement_factor)
    except KeyError:
        logger.debug(
            "Missing 'refinement_factor' key in 'simulation_params'. Defaulting to 1."
        )
        simulation_params["refinement_factor"] = 1
    except ValueError as e:
        err = f"Invalid value for 'refinement_factor' in 'simulation_params', got {refinement_factor}. Must be an integer."
        logger.error(err)
        raise ValueError(err) from e
    logger.debug(
        "Simulation parameters read successfully: {simulation_params}",
        simulation_params=simulation_params,
    )

    logger.debug("Reading digits_precision key...")
    try:
        digits_precision = data.pop("digits_precision")
        digits_precision = int(digits_precision)
        if backend == consts.BACKEND_FORTRAN or backend == consts.BACKEND_AUTO:
            logger.warning(
                "digits_precision is specified in the YAML file, but it will be ignored for Fortran backends."
            )
            digits_precision = consts.COMPUTE_PRECISION
    except KeyError:
        logger.debug(
            "Missing 'digits_precision' key in YAML file, but it is not required. Defaulting to consts.COMPUTE_PRECISION."
        )
        digits_precision = consts.COMPUTE_PRECISION
    except ValueError as e:
        err = f"Invalid value for 'digits_precision' in YAML file, got {digits_precision}. Must be an integer."
        logger.error(err)
        raise ValueError(err) from e
    logger.debug(
        "Digits precision read successfully: {digits_precision}",
        digits_precision=digits_precision,
    )

    has_wavefield = "spherical_wavefield" in data

    logger.debug("Reading receivers key...")
    try:
        receivers_read = data.pop("receivers")
    except KeyError:
        # Only warn about missing receivers when no other output is requested; a
        # wavefield-only run legitimately has no seismogram receivers.
        if has_wavefield:
            logger.info(
                "Missing 'receivers' key in YAML file, but a 'spherical_wavefield' block is present. Defaulting to empty list of seismogram locations."
            )
        else:
            logger.warning(
                "Missing 'receivers' key in YAML file. Defaulting to empty list of seismogram locations."
            )
        receivers_read = {"network": "AA", "receiver_list": []}
    receivers = {}
    try:
        network = receivers_read.pop("network")
        receivers["network"] = str(network)
    except KeyError:
        logger.debug(
            "Missing 'network' key in 'receivers', but it is not required. Defaulting to 'AA'."
        )
        receivers["network"] = "AA"
    try:
        receiver_list_read = receivers_read.pop("receiver_list")
        if not isinstance(receiver_list_read, list):
            err = f"Invalid value for 'receiver_list' in 'receivers', got {receiver_list_read}. Must be a list."
            logger.error(err)
            raise ValueError(err)
    except KeyError:
        logger.warning(
            "Missing 'receiver_list' key in 'receivers'. Defaulting to empty list."
        )
        receiver_list_read = []
    receivers["receiver_list"] = []
    for i, receiver_read in enumerate(receiver_list_read):
        receiver = {}
        try:
            receiver_location = receiver_read.pop("location")
            if not isinstance(receiver_location, list):
                err = f"Invalid value for 'location' in receiver entry {i + 1}, got {receiver_location}. Must be a list."
                logger.error(err)
                raise ValueError(err)
            if len(receiver_location) != dimension:
                err = f"Receiver location {receiver_location} has length {len(receiver_location)}, but dimension is {dimension}."
                logger.error(err)
                raise ValueError(err)
            receiver["location"] = np.array(
                [float(v) for v in receiver_location], dtype=float
            )
        except KeyError as e:
            err = f"Missing 'location' key in receiver entry {i + 1}."
            logger.error(err)
            raise KeyError(err) from e
        try:
            receiver_name = receiver_read.pop("name")
            receiver["name"] = str(receiver_name)
        except KeyError:
            logger.debug(
                "Missing 'name' key in receiver entry {n}, but it is not required. Defaulting to auto-generated name.",
                n=i + 1,
            )
            receiver["name"] = f"S{i + 1:04d}"
        try:
            seismogram_type = receiver_read.pop("seismogram_type")
            if not isinstance(seismogram_type, list):
                err = f"Invalid value for 'seismogram_type' in receiver entry {i + 1}, got {seismogram_type}. Must be a list."
                logger.error(err)
                raise ValueError(err)
            seismogram_types = set()
            for stype in seismogram_type:
                if stype == "displacement":
                    seismogram_types.add(consts.SEISMOGRAM_TYPE_DISPLACEMENT)
                elif stype == "rotation":
                    seismogram_types.add(consts.SEISMOGRAM_TYPE_ROTATION)
                else:
                    err = f"Invalid seismogram_type {stype} in receiver entry {i + 1}. Must be either 'displacement' or 'rotation'."
                    logger.error(err)
                    raise ValueError(err)
            receiver["seismogram_type"] = seismogram_types
        except KeyError:
            logger.debug(
                "Missing 'seismogram_type' key in receiver entry {n}, but it is not required. Defaulting to both displacement and rotation for Cosserat, only displacement for elastic.",
                n=i + 1,
            )
            receiver["seismogram_type"] = (
                {consts.SEISMOGRAM_TYPE_DISPLACEMENT, consts.SEISMOGRAM_TYPE_ROTATION}
                if material_type_code == consts.MATERIAL_TYPE_COSSERAT
                else {consts.SEISMOGRAM_TYPE_DISPLACEMENT}
            )
        if receiver_read:
            logger.warning(
                f"Extra keys in receiver entry {i + 1} that are not required or recognized: {', '.join(receiver_read.keys())}. They will be ignored."
            )
        receivers["receiver_list"].append(receiver)
    if receivers_read:
        logger.warning(
            f"Extra keys in 'receivers' that are not required or recognized: {', '.join(receivers_read.keys())}. They will be ignored."
        )
    logger.debug("Receivers read successfully: {receivers}", receivers=receivers)

    logger.debug("Reading spherical_wavefield key...")
    if "spherical_wavefield" in data:
        if dimension_code != consts.DIMENSION_3D:
            msg = f"'spherical_wavefield' output is 3D only, but dimension is {dimension}."
            logger.error(msg)
            raise ValueError(msg)
        spherical_wavefield = _parse_spherical_wavefield(
            data.pop("spherical_wavefield"), sources
        )
        logger.debug(
            "Spherical wavefield read successfully: {spherical_wavefield}",
            spherical_wavefield=spherical_wavefield,
        )
    else:
        logger.debug(
            "Missing 'spherical_wavefield' key in YAML file. No wavefield output will be written."
        )
        spherical_wavefield = None

    if data:
        logger.warning(
            f"Extra keys in YAML file that are not required or recognized: {', '.join(data.keys())}. They will be ignored."
        )

    return (
        dimension_code,
        material_type_code,
        backend_code,
        material_params,
        sources,
        simulation_params,
        digits_precision,
        receivers,
        spherical_wavefield,
        data_copy,
    )
