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
import pandas as pd
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQAlgorithm, VQE, MinimumEigensolver

from .energy_surface_spline import EnergySurfaceBase
from .extrapolator import Extrapolator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.drivers.molecule import Molecule

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
                 extrapolators: Optional[List[Extrapolator]] = None) -> None:
        """
        Args:
            gsc: GroundStateCalcualtion
            driver: BaseDriver
            tolerance: Tolerance desired for minimum energy.
            bootstrap: Whether to warm-start the solve of variational minimum eigensolvers.
            num_bootstrap: Number of previous points for extrapolation
                and bootstrapping. If None and a list of extrapolators is defined,
                all prev points will be used except the first two points will be used for
                bootstrapping. If no extrapolator is defined and bootstrap is True,
                all previous points will be used for bootstrapping.
            extrapolators: Extrapolator objects that define space/window and method to extrapolate
                variational parameters. First and second elements refer to the wrapper and internal
                extrapolators

        Raises:
            AquaError: If ``num_boostrap`` is an integer smaller than 2.
        """

        #TODO add a check the driver has a molecule

        self._gsc = gsc
        self._driver = driver
        self._tolerance = tolerance
        self._bootstrap = bootstrap
        self._results = None  # minimal DataFrame of [points, energies]
        self._results_full = None  # whole dict-of-dict-of-results
        self._points_optparams = None
        self._num_bootstrap = num_bootstrap
        self._extrapolator_wrap = None

        # set wrapper and internal extrapolators
        if extrapolators:
            # todo: assumed len(extrapolators) == 2
            self._extrapolator_wrap = extrapolators[0]  # wrapper
            self._extrapolator_wrap.extrapolator = extrapolators[1]  # internal extrapolator
            # set default number of bootstrapping points to 2
            if num_bootstrap is None:
                self._num_bootstrap = 2
                self._extrapolator_wrap.window = 0
            elif num_bootstrap >= 2:
                self._num_bootstrap = num_bootstrap
                self._extrapolator_wrap.window = num_bootstrap  # window for extrapolator
            else:
                raise AquaError(
                    'num_bootstrap must be None or an integer greater than or equal to 2')

        if isinstance(self._gsc.solver, VQAlgorithm):
            # Save initial point passed to min_eigensolver;
            # this will be used when NOT bootstrapping
            self._initial_point = self._gsc.solver.initial_point

    def run(self, points: List[float]) -> Tuple:
        """Run the sampler at the given points, potentially with repetitions.

        Args:
            points: The points along the degrees of freedom to evaluate.
            reps: Number of independent repetitions of this overall calculation.

        Returns:
            The results as pandas dataframe.
        """

        results_full = self.run_points(points)

        return results_full

    def run_points(self, points: List[float]) :
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
            result = self._run_single_point(point)  # execute single point here
            results_full[point] = result

        return results_full

    def _run_single_point(self, point: float) -> dict:
        """Run the sampler at the given single point

        Args:
            point: The value of the degree of freedom to evaluate.

        Returns:
            Results for a single point.
        """
        self._driver.molecule.perturbations = [point]

        print('Initial params , point: ', self._gsc.solver.initial_point, point)
        # find closest previously run point and take optimal parameters
        if isinstance(self._gsc.solver, VQAlgorithm) and self._bootstrap:
            prev_points = list(self._points_optparams.keys())
            print(prev_points)
            prev_params = list(self._points_optparams.values())
            n_pp = len(prev_points)

            if n_pp > 0:
                self._gsc.solver.initial_point = prev_params[0]

        # set number of points to bootstrap
            # if self._extrapolator_wrap is not None:
            #     n_boot = len(prev_points)  # bootstrap all points
            # else:
            #     n_boot = self._num_bootstrap

            #Set initial params if prev_points not empty
            # if prev_points:
            #     if n_pp <= n_boot:
            #         print('n_boot', n_boot)
            #         print('n_pp', n_pp)
            #         distances = np.array(point) - \
            #                     np.array(prev_points).reshape(n_pp, -1)
            #         # find min 'distance' from point to previous points
            #         min_index = np.argmin(np.linalg.norm(distances, axis=1))
            #         print('min_index', min_index)
            #         # update initial point
            #         self._gsc.solver.initial_point = prev_params[min_index]
            #     else:  # extrapolate using saved parameters
            #         opt_params = self._points_optparams
            #         param_sets = self._extrapolator_wrap.extrapolate(points=[point],
            #                                                          param_dict=opt_params)
            #         # update initial point, note param_set is a list
            #         self._gsc.solver.initial_point = param_sets.get(point)  # param set is a dictionary

            # if prev_points:
            #     distances = np.array(point) - np.array(prev_points).reshape(n_pp, -1)
            #     min_index = np.argmin(np.linalg.norm(distances, axis=1))
            #
            #     #print('min_index', min_index)
            #     # update initial point
            #     self._gsc.solver.initial_point = prev_params[-1]

        results = self._gsc.compute_groundstate(self._driver)

        # Save optimal point to bootstrap
        if isinstance(self._gsc.solver, VQAlgorithm):
            # at every point evaluation, the optimal params are updated
            optimal_params = self._gsc.solver.optimal_params
            self._points_optparams[point] = optimal_params

        print('Optimal Params, point: ', self._gsc.solver.optimal_params, point)

        return results

    # def _resampler(self) -> Tuple[float, int]:
    #     """Resample energy to mitigate sampling error/other noise.
    #
    #     Will re-evaluate energy enough times to get standard deviation below ``self._tolerance``.
    #
    #     Returns:
    #         A tuple containing the resampled energy and the number of additional evaluations made.
    #
    #     Raises:
    #         TypeError: If the min_eigensolver is not the VQE.
    #         AquaError: If there's a mismatch in the objective energy and the the mean of the
    #             callback.
    #     """
    #     # I only know how to make this work with VQE
    #     if not isinstance(self._min_eigensolver, VQE):
    #         raise TypeError('Currently only the VQE is handled as minimum eigensolver.')
    #         # logger.info("NOT resampling (minimum eigensolver is not VQE)")
    #         # return
    #
    #     optimal_parameters = self._min_eigensolver.optimal_params
    #
    #     # resampling is better if we can use a callback;
    #     callback_preserver = {
    #         'eval_count': None,
    #         'params': None,
    #         'mean': None,
    #         'std': None}
    #
    #     def callback(eval_count, params, mean, std):
    #         callback_preserver['eval_count'] = eval_count
    #         callback_preserver['params'] = params
    #         callback_preserver['mean'] = mean
    #         callback_preserver['std'] = std
    #
    #     original_shots = self._min_eigensolver.quantum_instance.run_config.shots
    #     original_callback = self._min_eigensolver._callback
    #     self._min_eigensolver._callback = callback
    #
    #     # Evaluate energy one more time, at optimal parameters
    #     # and get back standard deviation estimate (from callback)
    #     # Calculate how many times we need to re-sample the objective
    #     # in order to get an objective estimate with std deviation below desired tolerance
    #     # (Averaging objective n times has variance (objective variance)/n)
    #     extra_evals = 1
    #     objective_val = self._min_eigensolver._energy_evaluation(optimal_parameters)
    #     n_repeat = (callback_preserver['std'] / self._tolerance) ** 2
    #     if not np.isclose(objective_val, callback_preserver['mean'], 1e-7):
    #         raise AquaError("Mismatch in objective/energy in callback")
    #
    #     logger.info("Objective std dev: %s, repeats: %.2f", callback_preserver['std'], n_repeat)
    #
    #     #        oval = []
    #     #        for i in range(40):
    #     #            oval.append(self._min_eigensolver._energy_evaluation(optimal_parameters))
    #     #        print("Empirical std: {}".format(np.sqrt(np.var(oval, ddof=1))))
    #     #        print("Calced    std: {}".format(callback_preserver['std']))
    #
    #     if n_repeat > 1:
    #         total_shots = int(n_repeat * original_shots)
    #         # System limits;
    #         # max_shots = 8192 is a hard shot limit for hardware
    #         # max_reps controls total size of job/circuits sent
    #         #   (depending on circuit complexity may be larger/smaller)
    #         max_shots = 8192
    #         max_reps = 128
    #         total_shots = min(total_shots, max_shots * max_reps)
    #         rounded_evals = np.round(total_shots / original_shots, decimals=2)
    #         extra_evals += rounded_evals
    #         logger.info("Resampling objective %s times", rounded_evals)
    #         # Shot limit per call is 8192 (max_shots),
    #         # so break up total shots in some number of chunks.
    #         # If total shots is exactly divisible by 8192, great! what luck.
    #         # If not, take the ceiling of the quotient -
    #         # thats the number of chunks we'd have to do with at most 8192 shots each.
    #         # Then determine shots per chunk for that number of chunks we'd
    #         # have to do anyway
    #         n_repeat_chunk = np.ceil(total_shots / max_shots)
    #         chunk_shots = int(total_shots / n_repeat_chunk)
    #         rep_param = np.repeat(np.reshape(
    #             optimal_parameters, (1, -1)), n_repeat_chunk, axis=0).reshape(-1)
    #         # Update shot count for resampling
    #         self._min_eigensolver.quantum_instance.set_config(shots=chunk_shots)
    #         # Final return value is mean of all function evaluations
    #         objective_val = np.mean(self._min_eigensolver._energy_evaluation(rep_param))
    #         # Note that callback_preserver['eval_count'] counts this last
    #         # call as "one" evaluation
    #     # else
    #     # std deviation already below desired tolerance
    #     # use the value already calculated
    #     resampled_energy = objective_val
    #
    #     # set things back to normal
    #     self._min_eigensolver._callback = original_callback
    #     self._min_eigensolver.quantum_instance.set_config(shots=original_shots)
    #     return resampled_energy, extra_evals

    def fit_to_surface(self, energy_surface: EnergySurfaceBase, dofs: List[int],
                       **kwargs) -> None:
        """Fit the sampled energy points to the energy surface.

        Args:
            energy_surface: An energy surface object.
            dofs: A list of the degree-of-freedom dimensions to use as the independent
                variables in the potential function fit.
            **kwargs: Arguments to pass through to the potential's ``fit_to_data`` function.
        """
        points_all_dofs = self._results['point'].to_numpy()
        if len(points_all_dofs.shape) == 1:
            points = points_all_dofs.tolist()
        else:
            points = points_all_dofs[:, dofs].tolist()

        energies = self._results['energy'].to_list()
        energy_surface.fit_to_data(xdata=points, ydata=energies, **kwargs)
