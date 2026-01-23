from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SourceSpectrum(ABC):
    """
    ABC for a source time function in the frequency domain.

    The class provides the continuous Fourier transform of a source time function, defined by the convention
    f_hat(omega) = int f(t) exp(+i omega t) dt
    """

    @abstractmethod
    def spectrum(self, omega: float) -> complex:
        """
        The continuous Fourier transform of a source time function, defined by the convention
        f_hat(omega) = int f(t) exp(+i omega t) dt

        Parameters:
            omega (float): The frequency at which to evaluate the spectrum.

        Returns:
            complex: The Fourier transform of the STF.
        """
        ...

    @abstractmethod
    def direction(self) -> np.ndarray:
        """
        Get the source direction vector based on the specified angle.

        Returns:
            np.ndarray: A 3-element array representing the source direction vector.
        """
        ...

    def t0(self) -> float:
        """
        The time at which the seismogram trace should begin.

        Returns:
            float: The start time. Default implementation returns 0.
        """
        return 0.0
