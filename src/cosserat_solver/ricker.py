from __future__ import annotations

import numpy as np


class Ricker:
    """
    Ricker wavelet source time function.

    Parameters:
    f0: float
        The dominant frequency of the Ricker wavelet in Hz.
    f: float
        The displacement ratio.
    fc: float
        The rotation ratio.
    angle: float
        The source angle in degrees.
    factor: float
        A scaling factor for the amplitude.
    """

    def __init__(self, ricker_params: dict):
        self.f0 = ricker_params.get("f0", 25.0)  # in Hz
        self.omega0 = 2 * np.pi * self.f0
        self.displacement_ratio = ricker_params.get("f", 1.0)
        self.rotation_ratio = ricker_params.get("fc", 1.0)
        self.angle = ricker_params.get("angle", 0.0)  # in degrees
        self.factor = ricker_params.get("factor", 1.0)

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
