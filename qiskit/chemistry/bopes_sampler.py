# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The calculation of points on the Born-Oppenheimer Potential Energy Surface (BOPES)."""

import logging
from typing import Optional, Tuple, List, Dict

import numpy as np
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQAlgorithm
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.results.bopes_sampler_result import BOPESSamplerResult
from .energy_surface_spline import EnergySurfaceBase
from .extrapolator import Extrapolator

logger = logging.getLogger(__name__)


class BOPESSampler:
    """Class to evaluate the Born-Oppenheimer Potential Energy Surface (BOPES).
    """

    def __init__(self,
                 gsc: GroundStateCalculation,
                 driver: BaseDriver,
                 tolerance: float = 1e-3,
                 bootstrap: bool = True,
                 num_bootstrap: Optional[int] = None,
                 extrapolator: Optional[Extrapolator] = None) -> None:
        """
        Args:
            gsc: GroundStateCalculation
            driver: BaseDriver
            tolerance: Tolerance desired for minimum energy.
            bootstrap: Whether to warm-start the solve of variational minimum eigensolvers.
            num_bootstrap: Number of previous points for extrapolation
                and bootstrapping. If None and a list of extrapolators is defined,
                the first two points will be used for bootstrapping.
                If no extrapolator is defined and bootstrap is True,
                all previous points will be used for bootstrapping.
            extrapolator: Extrapolator objects that define space/window
                           and method to extrapolate variational parameters.

        Raises:
            AquaError: If ``num_boostrap`` is an integer smaller than 2.
        """

        # TODO add a check the driver has a molecule
        # TODO move driver to compute_surface

        self._gsc = gsc
        self._driver = driver
        self._tolerance = tolerance
        self._bootstrap = bootstrap
        self.results = dict()  # list of Tuples of (points, energies)
        self.results_full = None  # type: Optional[Dict]
        self._points_optparams = None  # type: Optional[Dict]
        self._num_bootstrap = num_bootstrap
        self._extrapolator = extrapolator

        if extrapolator:
            if num_bootstrap is None:
                # set default number of bootstrapping points to 2
                self._num_bootstrap = 2
                # self._extrapolator.window = 0
            elif num_bootstrap >= 2:
                self._num_bootstrap = num_bootstrap
                self._extrapolator.window = num_bootstrap  # window for extrapolator
            else:
                raise AquaError(
                    'num_bootstrap must be None or an integer greater than or equal to 2')

        if isinstance(self._gsc.solver, VQAlgorithm):
            # Save initial point passed to min_eigensolver;
            # this will be used when NOT bootstrapping
            self._initial_point = self._gsc.solver.initial_point

    def compute_surface(self, points: List[float]) -> BOPESSamplerResult:
        """Run the sampler at the given points, potentially with repetitions.

        Args:
            points: The points along the degrees of freedom to evaluate.

        Returns:
            BOPES Sampler Result
        """

        if self._driver.molecule is None:
            raise NotImplementedError('Please provide a molecule')

        if self._driver.molecule._degrees_of_freedom is None:
            raise NotImplementedError('Please provide dof in the molecule')

        # full dictionary of points
        self.results_full = self.run_points(points)
        # create results dictionary with (point, energy)
        self.results['point'] = list(self.results_full.keys())
        energies = []
        for key in self.results_full:
            energy = self.results_full[key]['computed_electronic_energy'] + \
                     self.results_full[key]['nuclear_repulsion_energy']
            energies.append(energy)
        self.results['energy'] = energies

        BOPESresult = BOPESSamplerResult(self.results, self.results_full)
        
        return BOPESresult

    def run_points(self, points: List[float]) -> Dict:
        """Run the sampler at the given points.

        Args:
            points: the points along the single degree of freedom to evaluate

        Returns:
            The results for all points.
        """
        results_full = dict()
        if isinstance(self._gsc.solver, VQAlgorithm):
            self._points_optparams = dict()
            self._gsc.solver.initial_point = self._initial_point

        # Iterate over the points
        for i, point in enumerate(points):
            logger.info('Point %s of %s', i + 1, len(points))
            result_full = self._run_single_point(point)  # dict of results
            results_full[point] = result_full
            # results.append([result_full['point'], result_full['energy']])

        return results_full

    def _run_single_point(self, point: float) -> dict:
        """Run the sampler at the given single point

        Args:
            point: The value of the degree of freedom to evaluate.

        Returns:
            Results for a single point.
        """

        # update molecule geometry and  thus resulting Hamiltonian based on specified point
        self._driver.molecule.perturbations = [point]

        # find closest previously run point and take optimal parameters
        if isinstance(self._gsc.solver, VQAlgorithm) and self._bootstrap:
            prev_points = list(self._points_optparams.keys())
            prev_params = list(self._points_optparams.values())
            n_pp = len(prev_points)

            # set number of points to bootstrap
            if self._extrapolator is None:
                n_boot = len(prev_points)  # bootstrap all points
            else:
                n_boot = self._num_bootstrap

            # Set initial params # if prev_points not empty
            if prev_points:
                if n_pp <= n_boot:
                    distances = np.array(point) - \
                                np.array(prev_points).reshape(n_pp, -1)
                    # find min 'distance' from point to previous points
                    min_index = np.argmin(np.linalg.norm(distances, axis=1))
                    # update initial point
                    self._gsc.solver.initial_point = prev_params[min_index]
                else:  # extrapolate using saved parameters
                    opt_params = self._points_optparams
                    param_sets = self._extrapolator.extrapolate(points=[point],
                                                                param_dict=opt_params)
                    # update initial point, note param_set is a list
                    # param set is a dictionary
                    self._gsc.solver.initial_point = param_sets.get(point)

        results = dict(self._gsc.compute_groundstate(self._driver))
        # Save optimal point to bootstrap
        if isinstance(self._gsc.solver, VQAlgorithm):
            # at every point evaluation, the optimal params are updated
            optimal_params = self._gsc.solver.optimal_params
            self._points_optparams[point] = optimal_params

        return results

    def fit_to_surface(self, energy_surface: EnergySurfaceBase, dofs: List[int],
                       **kwargs) -> None:
        """Fit the sampled energy points to the energy surface.

        Args:
            energy_surface: An energy surface object.
            dofs: A list of the degree-of-freedom dimensions to use as the independent
                variables in the potential function fit.
            **kwargs: Arguments to pass through to the potential's ``fit_to_data`` function.
        """
        points = self.results['point']
        energies = self.results['energy']
        energy_surface.fit_to_data(xdata=points, ydata=energies, **kwargs)

