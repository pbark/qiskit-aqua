# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
This module implements a 1D Harmonic potential. It can be used to compute
thermodynamic properties of diatomic molecules (e.g. H2, HD, D2) via the
diatomic partition function and the thermodynamics module.
"""
import numpy as np
from scipy.optimize import curve_fit

import qiskit.chemistry.constants as const
from .potential_base import PotentialBase


class HarmonicPotential(PotentialBase):
    """
    Implements a 1D Harmonic potential.
    Input units are Angstroms (distance between the two atoms),
        and output units are Hartrees (molecular energy).
    """
    # Works in Angstroms (input) and Hartrees (output)

    def __init__(self, molecule):
        """
        Constructor.
        Initializes the potential to the zero-function.
        fit_to_data() should be used afterwards to fit the potential to
            computed molecular energies.

        Args:
            molecule: the underlying molecule.

        Raises:
            ValueError: Only implemented for diatomic molecules
        """
        super().__init__(molecule)
        # Initialize with zero-potential.
        # Later - fit energy values (fit_to_data)
        self.k = 0.0
        self.m_shift = 0.0
        self.r_0 = 0.0
        self.d_e = None
        self._mA = molecule.masses[0]
        self._mB = molecule.masses[1]

    @staticmethod
    def fit_function(x, k, r_0, m_shift):
        """
        Functional form of the potential.
        """
        # K (Hartree/(Ang^2)), r_0 (Angstrom), m_shift (Hartree)
        return k / 2 * (x - r_0) ** 2 + m_shift

    def eval(self, x):
        """
        Evaluates the potential at a given point.
        """
        return self.fit_function(x, self.k, self.r_0, self.m_shift)

    def update_molecule(self, molecule):
        """
        Updates the underlying molecule.

        Raises:
            ValueError: Only implemented for diatomic molecules
        """
        # Check the provided molecule
        if len(molecule.masses) != 2:
            raise ValueError(
                'Harmonic potential only works for diatomic molecules!')
        self._mA = molecule.masses[0]
        self._mB = molecule.masses[1]

    def fit_to_data(self, xdata, ydata, initial_vals=None, bounds_list=None,
                    preprocess_data=True):
        """
        Fits a potential to computed molecular energies.

        Args:
            xdata: interatomic distance points (Angstroms)
            ydata: molecular energies (Hartrees)
            initial_vals: Initial values for fit parameters. None for default.
                    Order of parameters is k, r_0 and m_shift
                    (see fit_function implementation)
            bounds_list: Bounds for the fit parameters. None for default.
                    Order of parameters is k, r_0 and m_shift
                    (see fit_function implementation)
            preprocess_data: Default True. Internally cleans the data so
                that a fit is done only around the minimum. The wider range of
                distances and energies provided is still taken into account in
                estimating a dissociation energy for the molecule.
        """
        # Fits the potential to given x/y data.
        # If preProcessData is True (i.e. by default!) it only tries to fit
        # for values around the x where the minimum y was obtained.

        # do the Harmonic potential fit here, the order of parameters is
        # [k (Hartrees/(Ang**2)), r_0 (Ang), energy_shift (Hartrees)]
        h_p0 = (initial_vals if initial_vals is not None
                else np.array([0.2, 0.735, 1.5]))
        h_bounds = (bounds_list if bounds_list is not None
                    else ([0, -1, -2], [2, 3.0, 2]))

        xdata_fit = xdata
        ydata_fit = ydata
        if preprocess_data:
            xdata_fit, ydata_fit = HarmonicPotential.process_fit_data(
                xdata, ydata)

        fit, _ = curve_fit(self.fit_function, xdata_fit, ydata_fit,
                           p0=h_p0, maxfev=100000, bounds=h_bounds)

        self.k = fit[0]
        self.r_0 = fit[1]
        self.m_shift = fit[2]

        # Crude approximation of dissociation energy (assuming a reasonable
        # range of values were given)
        self.d_e = max(ydata) - min(ydata)

        # return

    def get_equilibrium_geometry(self, scaling=1.0):
        """
        Returns the interatomic distance corresponding to minimal energy.
        Args:
            scaling: Scaling to change units. (Default is 1.0 for Angstroms)
        """
        # the returned value(s) is defined by the molecule's degrees of
        # freedom?

        # Returns the distance for the minimal energy (scaled by 'scaling')
        # Default units (scaling=1.0) are Angstroms. Scale by 1E-10 to get
        # meters.
        return self.r_0 * scaling

    def get_minimal_energy(self, scaling=1.0):
        """
        Returns the smallest molecular energy for the current fit.
        Args:
            scaling: Scaling to change units. (Default is 1.0 for Hartrees)
        """
        # Returns the distance for the minimal energy (scaled by 'scaling')
        # Default units (scaling=1.0) are Hartrees. Scale appropriately for
        # Joules (per molecule or mol).
        return self.m_shift * scaling

    def dissociation_energy(self, scaling=1.0):
        """
        Returns the estimated dissociation energy for the current fit.
        Args:
            scaling: Scaling to change units. (Default is 1.0 for Hartrees)
        """
        # Hartree/(Ang**2) to Hartree!! 1/Ang**2 -> 1/m**2
        k = self.k * 1E20
        if self.d_e is not None:
            k = self.d_e

        diss_nrg = k - self.vibrational_energy_level(0)

        # in Hartree
        return diss_nrg * scaling

    def fundamental_frequency(self):
        """
        Returns the fundamental frequency for the current fit (in s^-1).
        """
        # Hartree(J)/(Ang**2), need Joules per molecule!! 1/Ang**2 -> 1/m**2
        k = self.k * const.HARTREE_TO_J * 1E20
        # r0 = self.r_0*1E-10  # angstrom, need meter
        mr = (self._mA * self._mB) / (self._mA + self._mB)

        # omega_0 in units rad/s converted to 1/s by dividing by 2Pi
        omega_0 = (np.sqrt(k / mr)) / (2 * np.pi)

        # fundamental frequency in s**-1
        return omega_0

    def wave_number(self):
        """
        Returns the wave number for the current fit (in cm^-1).
        """
        return self.fundamental_frequency() / const.C_CM_PER_S

    def vibrational_energy_level(self, n):
        """
        Returns the n-th vibrational energy level for the current fit
            (in Hartrees).
        """
        omega_0 = self.fundamental_frequency()
        e_n = const.H_J_S * omega_0 * (n + 0.5)

        # energy level
        return e_n * const.J_TO_HARTREE

    @classmethod
    def process_fit_data(cls, xdata, ydata):
        """
        Mostly for internal use. Preprocesses the data passed to fit_to_data()
            so that only the points around the minimum are fit (which gives
            more accurate vibrational modes).
        """
        sort_ind = np.argsort(xdata)
        ydata_s = ydata[sort_ind]
        xdata_s = xdata[sort_ind]
        min_y = min(ydata_s)

        # array of indices for X for which Y is min
        x_min = np.where(ydata_s == min_y)[0]

        # array of indices where X is equal to the value for which Y
        # is minimum
        all_of_min = np.array([], dtype=int)
        for i in x_min:
            all_of_min = np.concatenate(
                (all_of_min, np.where(xdata_s == xdata_s[i])[0]))
        # array of indices where X is equal to the next smaller value
        left_of_min = []
        if min(all_of_min) > 0:
            left_of_min = np.where(
                xdata_s == xdata_s[min(all_of_min) - 1])[0]
        # array of indices where X is equal to the next bigger value
        right_of_min = []
        if max(all_of_min) < (xdata_s.size - 1):
            right_of_min = np.where(
                xdata_s == xdata_s[max(all_of_min) + 1])[0]
        # all those indices together are used for fitting a harmonic
        # potential (around the min)
        inds = np.concatenate((left_of_min, all_of_min, right_of_min))

        return xdata_s[inds], ydata_s[inds]
