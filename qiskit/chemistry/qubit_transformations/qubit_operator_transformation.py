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

"""Base class for transformation to qubit operators for chemistry problems"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional

from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import EigenstateResult


class QubitOperatorTransformation(ABC):
    """Base class for transformation to qubit operators for chemistry problems"""

    @abstractmethod
    def transform(self, driver: BaseDriver,
                  aux_operators: Optional[List[Any]] = None
                  ) -> Tuple[WeightedPauliOperator, List[WeightedPauliOperator]]:
        """Transformation from the ``driver`` to a qubit operator.

        Args:
            driver: A driver encoding the molecule information.
            aux_operators: Additional auxiliary operators to evaluate. Must be of type
                ``FermionicOperator`` if the qubit transformation is fermionic and of type
                ``BosonicOperator`` it is bosonic.

        Returns:
            A qubit operator and a dictionary of auxiliary operators.
        """
        raise NotImplementedError

    @abstractmethod
    def interpret(self, eigenstate_result: EigenstateResult) -> EigenstateResult:
        """Interprets an EigenstateResult in the context of this transformation.

        Args:
            eigenstate_result: an eigenstate result object.

        Returns:
            An "interpreted" eigenstate result.
        """
        raise NotImplementedError
