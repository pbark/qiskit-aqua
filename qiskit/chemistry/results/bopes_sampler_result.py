# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""BOPES Sampler result"""

from typing import List, Optional, Tuple, cast

import logging

from qiskit.aqua.algorithms import AlgorithmResult

logger = logging.getLogger(__name__)


class BOPESSamplerResult:
    """The BOPES Sampler result"""

    def __init__(self, results, results_full):
        self._results = results
        self._results_full = results_full

    @property
    def points(self) -> list:
        """ returns list of points"""
        return self._results.get('point')

    @property
    def energies(self) -> list:
        """ returns list of energies"""
        return self._results.get('energy')

    @property
    def full_results(self) -> dict:
        """ returns all results for all points"""
        return self._results_full

    def point_results(self, point) -> dict:
        """ returns all results for all points"""
        return self._results_full[point]
