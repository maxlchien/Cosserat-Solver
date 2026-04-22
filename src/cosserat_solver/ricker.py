from __future__ import annotations

import numpy as np

from cosserat_solver.source import SourceSpectrum


class Ricker2D(SourceSpectrum):
    """
    Ricker wavelet source time function for the 2D problem.

    Parameters:
        f0 (float):
            The dominant frequency of the Ricker wavelet in Hz.
        f (float):
            The displacement ratio.
        fc (float):
            The rotation ratio.
        angle (float):
            The source angle in degrees.
        factor (float):
            A scaling factor for the amplitude.
        start_time (float):
            The start time for the trace. Defaults to -1.2 / f0.
    """

    def __init__(self, ricker_params: dict):
        self.f0 = ricker_params.get("f0", 25.0)  # in Hz
        self.omega0 = 2 * np.pi * self.f0
        self.displacement_ratio = ricker_params.get("f", 1.0)
        self.rotation_ratio = ricker_params.get("fc", 1.0)
        self.angle = ricker_params.get("angle", 0.0)  # in degrees
        self.factor = ricker_params.get("factor", 1.0)

        self.start_time = -1.2 / self.f0 + float(ricker_params.get("tshift", 0.0))

    def spectrum(self, omega: float) -> complex:
        """
        Compute the frequency spectrum of the Ricker wavelet at angular frequency omega.

        Parameters:
        omega: float
            The angular frequency in radians per second.

        Returns:
        complex
            The complex amplitude of the Ricker wavelet at frequency omega.
        """

        return (
            self.factor
            * (omega**2)
            / (2.0 * np.pi ** (5 / 2) * self.f0**3)
            * np.exp(-(omega**2) / (4.0 * np.pi**2 * self.f0**2))
        )

    def direction(self) -> np.ndarray:
        """
        Get the source direction vector based on the specified angle.

        Returns:
        np.ndarray
            A 3-element array representing the source direction vector.
        """
        angle_rad = np.deg2rad(self.angle)
        dir_x = self.displacement_ratio * np.cos(angle_rad)
        dir_z = self.displacement_ratio * np.sin(angle_rad)
        dir_y = self.rotation_ratio

        return np.array([dir_x, dir_z, dir_y])

    def t0(self) -> float:
        """
        The time at which the seismogram trace should begin.

        For the Ricker source, defaults to -1.2 / f0 unless specified otherwise.

        Returns:
            float: The start time.
        """
        return self.start_time


class Ricker3D(SourceSpectrum):
    """
    Ricker wavelet source time function for the 3D problem.

    Parameters:
        f0 (float):
            The dominant frequency of the Ricker wavelet in Hz.
        factor: float
            A scaling factor for the amplitude.
        f (np.ndarray):
            The displacement scaling factor. Specified as an array
            [f_x, f_y, f_z] for the 3D case.
        fc (np.ndarray):
            The rotation scaling factor. Specified as an array
            [fc_x, fc_y, fc_z] for the 3D case.
        start_time (float):
            The start time for the trace. Defaults to -1.2 / f0.
    """

    def __init__(self, ricker_params: dict):
        self.f0 = float(ricker_params.get("f0", 25.0))  # in Hz
        self.omega0 = 2 * np.pi * self.f0
        self.factor = float(ricker_params.get("factor", 1.0))
        self.displacement_factors = np.array(
            [float(x) for x in ricker_params.get("f", [1.0, 1.0, 1.0])]
        )
        self.rotation_factors = np.array(
            [float(x) for x in ricker_params.get("fc", [1.0, 1.0, 1.0])]
        )

        self.start_time = -1.2 / self.f0 + float(ricker_params.get("tshift", 0.0))

    def spectrum(self, omega: float) -> complex:
        """
        Compute the frequency spectrum of the Ricker wavelet at angular frequency omega.

        Parameters:
        omega: float
            The angular frequency in radians per second.

        Returns:
        complex
            The complex amplitude of the Ricker wavelet at frequency omega.
        """

        return (
            self.factor
            * (omega**2)
            / (2.0 * np.pi ** (5 / 2) * self.f0**3)
            * np.exp(-(omega**2) / (4.0 * np.pi**2 * self.f0**2))
        )

    def direction(self) -> np.ndarray:
        """
        Get the source scaling vector.

        Returns:
        np.ndarray
            A 6-element array representing the source scaling vector.
        """
        f_x = self.displacement_factors[0]
        f_y = self.displacement_factors[1]
        f_z = self.displacement_factors[2]
        f_cx = self.rotation_factors[0]
        f_cy = self.rotation_factors[1]
        f_cz = self.rotation_factors[2]

        return np.array([f_x, f_y, f_z, f_cx, f_cy, f_cz])

    def t0(self) -> float:
        """
        The time at which the seismogram trace should begin.

        For the Ricker source, defaults to -1.2 / f0 unless specified otherwise.

        Returns:
            float: The start time.
        """
        return self.start_time
