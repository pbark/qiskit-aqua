"""
16 January 2020
"""
import unittest
import numpy as np
from functools import partial
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.circuit.library import RealAmplitudes
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.aqua.operators import PauliExpectation
from qiskit.aqua.components.optimizers import AQGD
from qiskit.chemistry.bopes_sampler import BOPESSampler
from qiskit.chemistry.drivers import Molecule, PySCFDriver
from qiskit.chemistry.qubit_transformations import FermionicTransformation
from qiskit.chemistry.ground_state_calculation import MinimumEigensolverGroundStateCalculation
from qiskit.chemistry.morse_potential import MorsePotential

class TestBOPES(unittest.TestCase):

    def test_h2_bopes_sampler(self):
        np.random.seed(100)

        # Molecule
        dof = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        m = Molecule(geometry=[['H', [0., 0., 1.]],
                               ['H', [0., 0.45, 1.]]],
                     degrees_of_freedom=[dof])

        ft = FermionicTransformation()
        driver = PySCFDriver(molecule=m)

        qubitop, aux_ops = ft.transform(driver)

        # Quantum Instance:
        shots = 1
        backend = 'statevector_simulator'
        quantum_instance = QuantumInstance(Aer.get_backend(backend), shots=shots)
        quantum_instance.run_config.seed_simulator = 50
        quantum_instance.compile_config['seed_transpiler'] = 50

        # Variational form
        I_state = HartreeFock(num_orbitals=ft._molecule_info['num_orbitals'],
                              qubit_mapping=ft._qubit_mapping,
                              two_qubit_reduction=ft._two_qubit_reduction,
                              num_particles=ft._molecule_info['num_particles'],
                              sq_list=ft._molecule_info['z2_symmetries'].sq_list
                              )
        var_form = RealAmplitudes(qubitop.num_qubits, reps=1, entanglement='full',
                                  initial_state=I_state, skip_unentangled_qubits=False)


        # Classical optimizer:
        # Analytic Quantum Gradient Descent (AQGD) (with Epochs)
        aqgd_max_iter = [10] + [1] * 100
        aqgd_eta = [1e0] + [1.0 / k for k in range(1, 101)]
        aqgd_momentum = [0.5] + [0.5] * 100
        optimizer = AQGD(maxiter=aqgd_max_iter,
                         eta=aqgd_eta,
                         momentum=aqgd_momentum,
                         tol=1e-3,
                         averaging=4)

        # Min Eigensolver: VQE
        solver = VQE(var_form=var_form,
                     optimizer = optimizer,
                     quantum_instance=quantum_instance,
                     expectation=PauliExpectation())

        me_gsc = MinimumEigensolverGroundStateCalculation(ft, solver)

        # BOPES sampler
        bs = BOPESSampler(gsc=me_gsc, driver=driver)

        # absolute internuclear distance in Angstrom
        points = [0.7, 1.0, 1.3]
        results = bs.compute_surface(points)

        points_run = results.points
        energies = results.energies

        np.testing.assert_array_almost_equal(
            points_run, [0.7, 1.0, 1.3])
        np.testing.assert_array_almost_equal(
            energies, [-1.13618945, -1.10115033, -1.03518627], decimal=3)
        return

    def test_potential_interface(self):
        np.random.seed(100)

        stretch = partial(Molecule.absolute_distance, atom_pair=(1, 0))
        # H-H molecule near equilibrium geometry
        m = Molecule(geometry=[['H', [0., 0., 0.]],
                               ['H', [1., 0., 0.]],
                               ],
                     degrees_of_freedom=[stretch],
                     masses=[1.6735328E-27, 1.6735328E-27])

        ft = FermionicTransformation()
        driver = PySCFDriver(molecule=m)

        qubitop, aux_ops = ft.transform(driver)

        solver = NumPyMinimumEigensolver()

        me_gsc = MinimumEigensolverGroundStateCalculation(ft, solver)
        # Run BOPESSampler with exact eigensolution
        points = np.arange(0.45, 5.3, 0.3)
        bs = BOPESSampler(gsc=me_gsc, driver=driver)

        results = bs.compute_surface(points)

        # Testing Potential interface
        pot = MorsePotential(m)
        bs.fit_to_surface(pot, 0)

        np.testing.assert_array_almost_equal([pot.alpha, pot.r_0],
                                             [2.235, 0.720], decimal=3)
        np.testing.assert_array_almost_equal([pot.d_e, pot.m_shift],
                                             [0.2107, -1.1419], decimal=3)
        return


if __name__ == "__main__":
    unittest.main()
