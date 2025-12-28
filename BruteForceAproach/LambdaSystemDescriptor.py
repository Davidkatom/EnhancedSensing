import numpy as np
import qutip as qt

class LambdaSystem:
    """
    A class representing a composite 3-qubit Lambda system (qubits 1, 2, and e).
    The Hamiltonian is defined as:
    H = (J1/4) * (1 - Z1)(1 - Ze) + (J2/4) * (1 - Z2)(1 - Ze)
    
    Qubit order in tensor product: |1> (x) |2> (x) |e>
    indices: 0, 1, 2
    """
    def __init__(self, J1, J2, initial_state=None):
        """
        Initialize the Lambda System.

        Args:
            J1 (float): Coupling constant between qubit 1 and e.
            J2 (float): Coupling constant between qubit 2 and e.
            initial_state (qutip.Qobj, optional): Initial state of the system. 
                                                  Defaults to ground state |000>.
        """
        self.J1 = J1
        self.J2 = J2
        
        # Dimensions are fixed for 3 qubits
        self.dims = [2, 2, 2]
        
        if initial_state is not None:
            if initial_state.dims[0] != self.dims:
                 raise ValueError(f"State dims {initial_state.dims[0]} do not match system dims {self.dims}")
            self.state = initial_state
        else:
            # Default to |000>
            self.state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))

    @property
    def hamiltonian(self):
        """
        Constructs the Hamiltonian of the system.
        H = J1/4 * P1 + J2/4 * P2, where:
        P1 = (I - Z1)(I - Ze)
        P2 = (I - Z2)(I - Ze)
        Note: (1-Z) is actually 2 * |1><1| projector, so (1-Z)/2 is |1><1|.
        Thus (1-Z1)(1-Ze)/4 = |1_1><1_1| (x) |1_e><1_e|.
        Let's construct it explicitly with Pauli Z.
        """
        id2 = qt.qeye(2)
        sigmaz = qt.sigmaz()
        
        # Operators for qubit 1 (idx 0), qubit 2 (idx 1), qubit e (idx 2)
        z1 = qt.tensor(sigmaz, id2, id2)
        z2 = qt.tensor(id2, sigmaz, id2)
        ze = qt.tensor(id2, id2, sigmaz)
        
        # Identity for full system
        ide = qt.tensor(id2, id2, id2)
        
        # Terms: (1 - Z)
        term1 = (ide - z1) * (ide - ze)
        term2 = (ide - z2) * (ide - ze)
        
        H = (self.J1 / 4.0) * term1 + (self.J2 / 4.0) * term2
        return H

    def apply_unitary(self, U, targets):
        """
        Apply a unitary operator to specific qubits.
        
        Args:
            U (qutip.Qobj): Single-qubit or Multi-qubit Unitary operator.
            targets (list of int): Indices of target qubits [0, 1, 2].
        """
        if not isinstance(U, qt.Qobj):
            raise TypeError("U must be a qutip.Qobj")

        # If U is single qubit, expand it
        # If U is multi-qubit matching targets length, expand it
        # We generally use qutip.gate_expand_1toN or gate_expand_2toN etc 
        # but manual tensor construction is safer for general N.
        
        # Construct full unitary
        ops = [qt.qeye(2)] * 3
        
        if U.dims[0] == [2] * len(targets):
             # U covers all targets, construct tensor manually?
             # Easier way: if it's single qubit, just put it in the list.
             # If it's multi-qubit entangled, we need more care.
             # For now, let's assume separate tensor expansion if standard methods fail,
             # but qutip has tools.
             pass
        
        # Robust method for single/multi qubit unitaries using qutip specific tools if possible,
        # or manual construction.
        # Let's support single qubit U on single target or multi-qubit U on contiguous targets easily.
        # Arbitrary connectivity requires swapping or careful tensor-ing.
        
        # Simple implementation: Tensor product logic
        # Warning: This simple "replace identity" logic only works for strictly local-ordered ops
        # or single qubit ops.
        
        full_U = None
        
        num_targets = len(targets)
        if U.shape == (2**num_targets, 2**num_targets):
             # General case helper
             full_U = self._expand_unitary(U, targets)
        else:
             raise ValueError("Dimension of U does not match number of targets")
             
        self.state = full_U * self.state
        if self.state.isket:
             # Ensure normalization? unitary preserves it.
             pass
        elif self.state.isoper:
             self.state = full_U * self.state * full_U.dag()

    def _expand_unitary(self, U, targets):
        """Helper to expand unitary to full 3-qubit space"""
        # Sort targets to handle them in order if we construct via tensor?
        # Actually, qutip doesn't have a generic 'expand_any' that is easy 
        # for arbitrary targets without building it.
        
        # Basic approach:
        # If single target: tensor(I, U, I)
        # If multiple targets: more complex.
        
        if len(targets) == 1:
            t = targets[0]
            ops = [qt.qeye(2), qt.qeye(2), qt.qeye(2)]
            ops[t] = U
            return qt.tensor(ops)
            
        elif len(targets) == 2:
             # e.g. targets=[0, 2]
             # If U is a 4x4 matrix, we need to decompose or map indices.
             # However, simpler is if U is separable, but usually it's CNOT etc.
             # We can use qutip implementation of specific gates or specialized expansion.
             # But here we need generic U.
             
             # Reverting to explicit matrix indexing for ordering [0, 1, 2]
             # This is complex to implement generically from scratch without checking qutip docs for `gate_expand_2toN` equivalent.
             # Let's look for qutip.tensor usage.
             pass

        # Fallback: Use qutip's superoperator/tensor support? 
        # Actually, let's iterate. 
        # For this specific task, let's assume mainly single qubit gates 
        # OR just handle all cases by manually constructing the operator matrix if needed? No, inefficient.
        
        # Best bet for robust generic code without complex dependency:
        # Only support 1-qubit unitaries for now, or 2-qubit if adjacent?
        # The prompt asks for "Apply unitary (both for specific qubit and multiqubit)".
        
        # Let's implement a generatic permute-based expansion.
        # 1. Expand U to full space assuming targets are [0, 1, ... k]
        # 2. Permute dimensions to match actual targets.
        
        # Full dims
        N = 3
        # U acts on targets.
        # Create a tensor of U (x) I (x) ...
        # Then permute.
        
        # Actually, we can assume U is given in the basis of the targets.
        # We need to map 'targets' to system indices.
        
        # Construct list of dimensions for the "pre-permuted" object
        # [2, 2, ... ] for targets, then [2, 2...] for rest.
        
        # This is getting complicated.
        # SIMPLIFICATION:
        # If len(targets) == 3: return U
        # If len(targets) == 1: Use basic tensor.
        # If len(targets) == 2:
        #   Use qutip arithmetic.
        
        # Let's try to stick to standard tensor creation if possible.
        # For [0, 2] (qubit 1 and e), we need to insert I at 1.
        # U_1e.
        # If U is 4x4, we can't just tensor it with I.
        # We need a permute.
        
        # Create operator on ordered system [t1, t2, rest...]
        # Then permute back to [0, 1, 2]
        
        system_dims = [2] * N
        
        # Determine complementary indices
        all_indices = list(range(N))
        complement = [i for i in all_indices if i not in targets]
        
        # Construct U_tensor = U (x) I_rest
        # This acts on Hilbert space H_targets (x) H_complement
        
        # The dimension list for U_tensor is [dims_targets, dims_complement]
        # We want to permute this to [dims_0, dims_1, dims_2]
        
        if len(complement) > 0:
             I_rest = qt.tensor([qt.qeye(2) for _ in complement])
             U_full_unordered = qt.tensor(U, I_rest)
        else:
             U_full_unordered = U
             
        # Current order of subsystems: targets + complement
        current_order = targets + complement
        
        # We need an inverse permutation to get back to [0, 1, 2]
        # order[i] = k means the i-th subsystem in current_order is the k-th physical qubit?
        # No. current_order[k] is the physical qubit index at position k.
        # We want to permute the state dimensions such that they align with physical 0, 1, 2.
        
        # qutip.Qobj.permute(order) rearranges the tensor structure.
        # The argument `order` specifies the new order of the *current* dimensions.
        # If current dimensions correspond to physical qubits [q_t1, q_t2, ... q_c1...],
        # we want to rearrange them so they are [q_0, q_1, q_2].
        
        # So we need to find the index in `current_order` that corresponds to physical 0, then 1, then 2.
        perm_order = [current_order.index(i) for i in range(N)]
        
        return U_full_unordered.permute(perm_order)

    def time_evolve(self, t_list, c_ops=[], args=None):
        """
        Evolve the system state according to H.
        
        Args:
            t_list (list/array): Time points for evolution.
            c_ops (list): List of collapse operators for Lindblad master equation.
            args (dict): Arguments for Hamiltonian coefficients (if time dependent).
            
        Returns:
            result: qutip.Result object containing states/expectations.
        """
        H = self.hamiltonian
        
        # If we have c_ops or the state is a density matrix, use operator evolution
        # otherwise schrodinger
        
        if c_ops or self.state.isoper:
             result = qt.mesolve(H, self.state, t_list, c_ops=c_ops, args=args)
        else:
             result = qt.mesolve(H, self.state, t_list, c_ops=[], args=args)
             
        # Update state to the last time point? 
        # Usually time_evolve returns the trajectory.
        # Let's update internal state to the final state for continuity?
        # User defined "methods like time_evolve", didn't specify side effects.
        # Standard object simulation often updates the state.
        self.state = result.states[-1]
        
        return result

if __name__ == "__main__":
    # Quick sanity check code block
    sys = LambdaSystem(J1=1.0, J2=1.0)
    print("Hamiltonian dims:", sys.hamiltonian.dims)
    print("Initial state:", sys.state)

