import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# =====================================================================
# Parameters
# =====================================================================
N = 10 # Number of bath spins
Omega = 10.0  # Transverse field on central spin
J = 1.0      # Interaction strength
t_max = 10                # Total simulation time 
n_steps = 500              # Time steps for calculation
tlist = np.linspace(0, t_max, n_steps)

# =====================================================================
# Operators (Dicke Basis for speedup)
# =====================================================================
# The N bath spins are symmetric, so we can use a single large spin S = N/2.
# This reduces the Hilbert space size from 2^(N+1) to 2 * (N+1).
S_spin = N / 2.0
dim_bath = int(2 * S_spin + 1)

# Bath spin operators (dimension: 2S + 1)
# Note: The sum of Pauli matrices over the bath equals 2 * the spin-S operator.
Jx = qt.jmat(S_spin, 'x') * 2.0
Jy = qt.jmat(S_spin, 'y') * 2.0
Jz = qt.jmat(S_spin, 'z') * 2.0
Is_bath = qt.qeye(dim_bath)

# Central spin operators (dimension: 2)
si = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

# Full Hilbert space operators (Central Spin \otimes Bath)
sx_s = qt.tensor(sx, Is_bath)
sy_s = qt.tensor(sy, Is_bath)
sz_s = qt.tensor(sz, Is_bath)

Sx = qt.tensor(si, Jx)
Sy = qt.tensor(si, Jy)
Sz = qt.tensor(si, Jz)

# =====================================================================
# Hamiltonians
# =====================================================================
# Note: Based on the Schrieffer-Wolff transformation H_eff = Omega*sx_s + (J^2/2Omega)*sx_s*Sz^2,
# the exact Hamiltonian corresponds to: H = Omega*sx_s + J*sz_s*Sz.
# (Assuming the prompt's 'JSzSigma_x' and 'Sz is sum of sigma_Xi' were typos for 'JSzSigma_z' and 'sigma_Zi', 
# since with X the interaction would be trivially diagonal or lead to a different effective Hamiltonian).
H1 = Omega * sx_s + J * sz_s * Sz

H2 = Omega * sx_s + (J**2 / (2 * Omega)) * sx_s * (Sz**2)

# =====================================================================
# Initial state
# =====================================================================
# Central spin in |+_s> (the +1 eigenstate of sigma_x)
plus_state_central = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

# Bath in |+>^N. In the Dicke basis, this is a spin coherent state pointing along +x
plus_state_bath = qt.spin_coherent(S_spin, np.pi/2, 0)

psi0 = qt.tensor(plus_state_central, plus_state_bath)

# =====================================================================
# Evolution
# =====================================================================
# Observables to calculate expectation values for bath spins
e_ops = [Sz, Sx, Sy, Sz**2, Sx**2, Sy**2]

res1 = qt.mesolve(H1, psi0, tlist, e_ops=e_ops)
res2 = qt.mesolve(H2, psi0, tlist, e_ops=e_ops)

# =====================================================================
# Plotting
# =====================================================================
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
fig.suptitle(fr'Bath Spin Dynamics (N={N}, $\Omega={Omega}$, $J={J}$)' + '\n' + r'Initial State: $|+>_s \otimes |+>^N$')

# Plot Bath Spin Dynamics
axs[0, 0].set_title("Expectation Values")
axs[0, 0].plot(tlist, res1.expect[0], label='Exact', linestyle='-')
axs[0, 0].plot(tlist, res2.expect[0], label='SW Eff', linestyle='--')
axs[0, 0].set_ylabel(r'$\langle S_z \rangle$')

axs[1, 0].plot(tlist, res1.expect[1], label='Exact', linestyle='-')
axs[1, 0].plot(tlist, res2.expect[1], label='SW Eff', linestyle='--')
axs[1, 0].set_ylabel(r'$\langle S_x \rangle$')

axs[2, 0].plot(tlist, res1.expect[2], label='Exact', linestyle='-')
axs[2, 0].plot(tlist, res2.expect[2], label='SW Eff', linestyle='--')
axs[2, 0].set_ylabel(r'$\langle S_y \rangle$')
axs[2, 0].set_xlabel('Time')

# Plot Bath Spin Squared Dynamics
axs[0, 1].set_title("Squared Observables")
axs[0, 1].plot(tlist, res1.expect[3], label='Exact', linestyle='-')
axs[0, 1].plot(tlist, res2.expect[3], label='SW Eff', linestyle='--')
axs[0, 1].set_ylabel(r'$\langle S_z^2 \rangle$')

axs[1, 1].plot(tlist, res1.expect[4], label='Exact', linestyle='-')
axs[1, 1].plot(tlist, res2.expect[4], label='SW Eff', linestyle='--')
axs[1, 1].set_ylabel(r'$\langle S_x^2 \rangle$')

axs[2, 1].plot(tlist, res1.expect[5], label='Exact', linestyle='-')
axs[2, 1].plot(tlist, res2.expect[5], label='SW Eff', linestyle='--')
axs[2, 1].set_ylabel(r'$\langle S_y^2 \rangle$')
axs[2, 1].set_xlabel('Time')

for ax in axs.flat:
    ax.legend(loc='best')
    ax.grid(True)

plt.tight_layout()
plt.show()
