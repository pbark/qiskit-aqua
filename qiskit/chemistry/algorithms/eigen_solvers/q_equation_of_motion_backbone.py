class EOMExcitedStatesCalculation:

    def __init__(self, ground_state_calculation, excitation_type, full_active_space = True):
        self._gsc = ground_state_calculation

        if [letter not in ['S','D','T'] for letter in excitation_type]:
            raise ValueError(
                'Excitation type must be S (single), D (double), T (triple) or a combination of them')

        self._excitation_type = excitation_type
        self._full_active_space = full_active_space
        """
        Constructor

        :param ground_state_calculation: a GroundStateCalculation object, contains all information
        about how the ground state should be prepared.
        :param excitation_type: 'S' (single), 'D' (double), 'T' (triple) or a combination of them
        e.g. 'SD', 'SDT'.
        :param full_active_space: if True all degrees of freedom will be considered for 
        constructing the EOM matrix elements. If False the active space will be restricted to that 
        of the ground state calculation. 
        """

    def calculate_excited_states(self, driver, quantum_instance):
        """
        construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients
        """
        results = {'excitation energies': None,
                   'expansion coefficients': None,
                   'm_mat': None, 'v_mat': None, 'q_mat': None, 'w_mat': None,
                   'm_mat_std': None, 'v_mat_std': None,
                   'q_mat_std': None, 'w_mat_std': None}

        return results


    def build_eom_matrices(self, ground_state, quantum_instance,
                                    q_commutators, w_commutators, m_commutators, v_commutators):

        """
        Compute the all matrix elements to reconstruc the EOM pseudo-eigenvalue problem
        :param ground_state: the ground state on which each operator is evaluated
        :param quantum_instance: the quantum instance for the evaluation of the operators
        :param q_commutators: operators for each element of the Q matrix
        :param w_commutators: operators for each element of the W matrix
        :param m_commutators: operators for each element of the M matrix
        :param v_commutators: operators for each element of the V matrix
        :return:
        """

        #...

        return m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std

    def compute_excitation_energies(self, m_mat, v_mat, q_mat, w_mat):
        """
        Classically solves the EOM pseudo-eigenvalue problem
        :param m_mat: M matrix
        :param v_mat: V matrix
        :param q_mat: Q matrix
        :param w_mat: W matrix
        :return:
        """

        #...

        # excitation_energies are the eigenvalues of the pseudo-eigenvalue problem
        # expansion_coefficients are the eigenvectors of the pseudo-eigenvalue problem

        return excitation_energies, expansion_coefficients

class NumericalEOMExcitedStatesCalculation(EOMExcitedStatesCalculation):

    def __init__(self, ground_state_calculation, excitation_type, full_active_space = True):
        super().__init__(ground_state_calculation, excitation_type,
                         full_active_space=full_active_space)


    def calculate_excited_states(self, driver, quantum_instance):

        """
        Calculate excitations energies
        :param driver: the classical driver, contains the molecular information
        :return: the excitation energies
        """

        ground_state = self._gsc.compute_ground_state(driver)

        if self._full_active_space:
            excitations_list = self._gsc.transformation.build_excitation_list(self._excitation_type)
        else:
            excitations_list = self._gsc.transformation.build_excitation_list(self._excitation_type,
                                                                              self._gsc.active_space)

        hopping_operators, type_of_commutativities = self._gsc.transformation.build_hopping_operators(
            excitations_list)

        q_commutators, w_commutators, m_commutators, v_commutators, available_entry = \
            self.build_all_commutators(excitations_list, hopping_operators, type_of_commutativities)

        m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std = \
            self.build_eom_matrices(ground_state, quantum_instance,
                                    q_commutators, w_commutators, m_commutators, v_commutators)

        excitation_energies, expansion_coefficients = self.compute_excitation_energies(
            m_mat, v_mat, q_mat, w_mat)

        results = {'excitation energies':  excitation_energies,
                   'expansion coefficients': expansion_coefficients,
                   'm_mat': m_mat, 'v_mat': v_mat, 'q_mat': q_mat, 'w_mat': w_mat,
                   'm_mat_std': m_mat_std, 'v_mat_std': v_mat_std,
                   'q_mat_std': q_mat_std, 'w_mat_std': w_mat_std}

        return results

    def build_all_commutators(self, excitations_list, hopping_operators, type_of_commutativities):
        """Building all commutators for Q, W, M, V matrices.

         Args:
             excitations_list (list): single excitations list + double excitation list
             hopping_operators (dict): all hopping operators based on excitations_list,
                                       key is the string of single/double excitation;
                                       value is corresponding operator.
             type_of_commutativities (dict): if tapering is used, it records the commutativities of
                                      hopping operators with the
                                      Z2 symmetries found in the original operator.
         Returns:
             dict: key: a string of matrix indices; value: the commutators for Q matrix
             dict: key: a string of matrix indices; value: the commutators for W matrix
             dict: key: a string of matrix indices; value: the commutators for M matrix
             dict: key: a string of matrix indices; value: the commutators for V matrix
             int: number of entries in the matrix
         """

        # ...

        return q_commutators, w_commutators, m_commutators, v_commutators, available_entry


class AnalyticalEOMExcitedStatesCalculation(EOMExcitedStatesCalculation):

    def __init__(self, ground_state_calculation, excitation_type, full_active_space = True):
        super().__init__(ground_state_calculation, excitation_type,
                         full_active_space=full_active_space)

    def calculate_excited_states(self, driver, quantum_instance):

        """
        Calculate excitations energies
        :param driver: the classical driver, contains the molecular information
        :return: the excitation energies
        """

        ground_state = self._gsc.compute_ground_state(driver)

        if self._full_active_space:
            excitations_list = self._gsc.transformation.build_excitation_list(self._excitation_type)
        else:
            excitations_list = self._gsc.transformation.build_excitation_list(self._excitation_type,
                                                                              self._gsc.active_space)

        ### Need to check how this part is replaced according to Mario's code
        hopping_operators, type_of_commutativities = self._gsc.transformation.build_hopping_operators(
            excitations_list)

        q_commutators, w_commutators, m_commutators, v_commutators, available_entry = \
            self.build_all_commutators(excitations_list, hopping_operators, type_of_commutativities)
        ###

        m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std = \
            self.build_eom_matrices(ground_state, quantum_instance,
                                    q_commutators, w_commutators, m_commutators, v_commutators)

        excitation_energies, expansion_coefficients = self.compute_excitation_energies(
            m_mat, v_mat, q_mat, w_mat)

        results = {'excitation energies': excitation_energies,
                   'expansion coefficients': expansion_coefficients,
                   'm_mat': m_mat, 'v_mat': v_mat, 'q_mat': q_mat, 'w_mat': w_mat,
                   'm_mat_std': m_mat_std, 'v_mat_std': v_mat_std,
                   'q_mat_std': q_mat_std, 'w_mat_std': w_mat_std}

        return results

    def build_all_commutators(self, arguments):
        """
        check with Mario's code
        """

        # ...

        return q_commutators, w_commutators, m_commutators, v_commutators, available_entry