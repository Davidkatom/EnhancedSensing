import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# =====================================================================
# Parameters
# =====================================================================
N = 10 # Number of bath spins
Omega = 20  # Transverse field on central spin
J = 1.0      # Interaction strength
gamma = 0.1 # Dephasing rate on central spin
beta = 0.0
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
# H2 = Omega * sx_s + J * sz_s * Sz

# =====================================================================
# Initial state
# =====================================================================
# Central spin in |+_s> (the +1 eigenstate of sigma_x)
plus_state_central = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
# plus_state_central = (qt.basis(2, 1)).unit()

# Bath in |+>^N. In the Dicke basis, this is a spin coherent state pointing along +x
plus_state_bath = qt.spin_coherent(S_spin, np.pi/2, 0)

psi0 = qt.tensor(plus_state_central, plus_state_bath)

# =====================================================================
# Evolution
# =====================================================================
# Dephasing collapse operator on the central spin
# c_ops = [np.sqrt(gamma) * sx_s]
c_ops = [np.sqrt(beta) * sx_s, np.sqrt(gamma) * sz_s]
# Observables to calculate expectation values for bath spins
e_ops = [Sz, Sx, Sy, Sz**2, Sx**2, Sy**2]

res1 = qt.mesolve(H1, psi0, tlist, c_ops=c_ops, e_ops=e_ops)
res2 = qt.mesolve(H2, psi0, tlist, c_ops=c_ops, e_ops=e_ops)

# =====================================================================
# Adiabatic Approximation
# =====================================================================
lambda_val = (J**2 / (2 * Omega)) if Omega != 0.0 else 0.0

rho0_bath = plus_state_bath * plus_state_bath.dag()
rho0_mat = rho0_bath.full()

# Dicke basis states m values (from N/2 down to -N/2)
m_vals = np.diag((qt.jmat(S_spin, 'z') * 2.0).full())
n_mat, m_mat = np.meshgrid(m_vals, m_vals, indexing='ij')

e_ops_bath = [Jz, Jx, Jy, Jz**2, Jx**2, Jy**2]
adiab_expect = np.zeros((len(e_ops_bath), len(tlist)))

for idx, t in enumerate(tlist):
    # Calculate r_{n,m}(t)
    r_nm = np.exp(-1 * lambda_val**2 * (n_mat**2 - m_mat**2)**2 / (2*gamma) * t - 1*(lambda_val/Omega) * gamma * (n_mat - m_mat)**2 * t)
    
    # Calculate rho_bath(t)
    rho_t_mat = rho0_mat * r_nm
    rho_t = qt.Qobj(rho_t_mat, dims=rho0_bath.dims)
    
    # Expectation values for the bath
    for op_idx, op in enumerate(e_ops_bath):
        adiab_expect[op_idx, idx] = np.real(qt.expect(op, rho_t))

# =====================================================================
# Analytical Solution 2
# =====================================================================
# =====================================================================
# Analytical Solution 2 - corrected equation
# =====================================================================
r0 = 1.0
x0 = 1.0

# Use eigenvalues consistent with your numerical Sz = sum sigma_z
# If your derivation uses collective J_z instead, replace Jz by qt.jmat(S_spin, 'z')
m_vals_sol2 = np.diag(Jz.full())
n_mat, m_mat = np.meshgrid(m_vals_sol2, m_vals_sol2, indexing='ij')


C1 = (
    2 * gamma
    + (J**2 * (beta + gamma) * (n_mat - m_mat)**2) / (2 * Omega**2)
    + (J**2 * beta * (m_mat + n_mat)**2) / (2 * Omega**2)
)


C0 = (
    (J**2 * gamma * (beta + gamma) * (n_mat - m_mat)**2) / (Omega**2)
    + (J**4 * (n_mat**2 - m_mat**2)**2) / (4 * Omega**2)
)

Delta = np.sqrt(C1**2 - 4 * C0 + 0j)

# Delta = 2*gamma * np.ones_like(n_mat)


sol2_expect = np.zeros((len(e_ops_bath), len(tlist)))

for idx, t in enumerate(tlist):
    cosh_term = np.cosh(Delta * t / 2)
    sinh_term = np.sinh(Delta * t / 2)

    numerator = (
        1j * J**2 * (n_mat**2 - m_mat**2) * x0 / Omega
        - (J**2 * (beta + gamma) * (n_mat - m_mat)**2 * r0) / (Omega**2)
        + C1 * r0
    )

    # Safe division by Delta
    safe_sinh_over_delta = np.zeros_like(Delta, dtype=np.complex128)
    nonzero_mask = np.abs(Delta) > 1e-12
    zero_mask = ~nonzero_mask

    safe_sinh_over_delta[nonzero_mask] = sinh_term[nonzero_mask] / Delta[nonzero_mask]

    # Since sinh(Delta*t/2)/Delta -> t/2 as Delta -> 0
    safe_sinh_over_delta[zero_mask] = t / 2.0

    r_nm_t = np.exp(-C1 * t / 2) * (
        r0 * cosh_term
        + numerator * safe_sinh_over_delta
    )

    rho_t_mat = rho0_mat * r_nm_t
    rho_t = qt.Qobj(rho_t_mat, dims=rho0_bath.dims)

    for op_idx, op in enumerate(e_ops_bath):
        sol2_expect[op_idx, idx] = np.real(qt.expect(op, rho_t))
# =====================================================================
# Two Spin Dynamics (Central + 1 Bath Spin)
# =====================================================================
# Simulate a single bath spin interacting with the central spin
sz_s_2q = qt.tensor(sz, si)
sz_1_2q = qt.tensor(si, sz)
sx_1_2q = qt.tensor(si, sx)
sy_1_2q = qt.tensor(si, sy)

H_2q = J * sz_s_2q * sz_1_2q
c_ops_2q = [np.sqrt(gamma) * sz_s_2q]
e_ops_2q = [sz_1_2q, sx_1_2q, sy_1_2q]

plus_state_1q = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
psi0_2q = qt.tensor(plus_state_1q, plus_state_1q) # Both in |+>

res_sq = qt.mesolve(H_2q, psi0_2q, tlist, c_ops=c_ops_2q, e_ops=e_ops_2q)

# =====================================================================
# Plotting
# =====================================================================
fig, axs = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
fig.suptitle(fr'Bath Spin Dynamics (N={N}, $\Omega={Omega}$, $J={J}$, $\gamma={gamma}$)' + '\n' + r'Initial State: $|+>_s \otimes |+>^N$')

# Plot Bath Spin Dynamics
axs[0, 0].set_title("Expectation Values")
axs[0, 0].plot(tlist, res1.expect[0], label='Exact', linestyle='-')
# axs[0, 0].plot(tlist, res2.expect[0], label='SW Eff', linestyle='--')
axs[0, 0].plot(tlist, adiab_expect[0], label='Adiabatic', linestyle=':')
axs[0, 0].plot(tlist, sol2_expect[0], label='Analytic 2', linestyle='-.')
axs[0, 0].set_ylabel(r'$\langle S_z \rangle$')

axs[1, 0].plot(tlist, res1.expect[1], label='Exact', linestyle='-')
# axs[1, 0].plot(tlist, res2.expect[1], label='SW Eff', linestyle='--')
axs[1, 0].plot(tlist, adiab_expect[1], label='Adiabatic', linestyle=':')
axs[1, 0].plot(tlist, sol2_expect[1], label='Analytic 2', linestyle='-.')
axs[1, 0].set_ylabel(r'$\langle S_x \rangle$')

axs[2, 0].plot(tlist, res1.expect[2], label='Exact', linestyle='-')
# axs[2, 0].plot(tlist, res2.expect[2], label='SW Eff', linestyle='--')
axs[2, 0].plot(tlist, adiab_expect[2], label='Adiabatic', linestyle=':')
axs[2, 0].plot(tlist, sol2_expect[2], label='Analytic 2', linestyle='-.')
axs[2, 0].set_ylabel(r'$\langle S_y \rangle$')
axs[2, 0].set_xlabel('Time')

# Plot Bath Spin Squared Dynamics
axs[0, 1].set_title("Squared Observables")
axs[0, 1].plot(tlist, res1.expect[3], label='Exact', linestyle='-')
# axs[0, 1].plot(tlist, res2.expect[3], label='SW Eff', linestyle='--')
axs[0, 1].plot(tlist, adiab_expect[3], label='Adiabatic', linestyle=':')
axs[0, 1].plot(tlist, sol2_expect[3], label='Analytic 2', linestyle='-.')
axs[0, 1].set_ylabel(r'$\langle S_z^2 \rangle$')

axs[1, 1].plot(tlist, res1.expect[4], label='Exact', linestyle='-')
# axs[1, 1].plot(tlist, res2.expect[4], label='SW Eff', linestyle='--')
axs[1, 1].plot(tlist, adiab_expect[4], label='Adiabatic', linestyle=':')
axs[1, 1].plot(tlist, sol2_expect[4], label='Analytic 2', linestyle='-.')
axs[1, 1].set_ylabel(r'$\langle S_x^2 \rangle$')

axs[2, 1].plot(tlist, res1.expect[5], label='Exact', linestyle='-')
# axs[2, 1].plot(tlist, res2.expect[5], label='SW Eff', linestyle='--')
axs[2, 1].plot(tlist, adiab_expect[5], label='Adiabatic', linestyle=':')
axs[2, 1].plot(tlist, sol2_expect[5], label='Analytic 2', linestyle='-.')
axs[2, 1].set_ylabel(r'$\langle S_y^2 \rangle$')
axs[2, 1].set_xlabel('Time')

# Plot Two Qubit Model (Bath Spin 1)
axs[0, 2].set_title("Single Bath Spin ($Z_s Z_1$)")
axs[0, 2].plot(tlist, res_sq.expect[0], label='Single Bath Spin', linestyle='-', color='g')
axs[0, 2].set_ylabel(r'$\langle Z_1 \rangle$')

axs[1, 2].plot(tlist, res_sq.expect[1], label='Single Bath Spin', linestyle='-', color='g')
axs[1, 2].set_ylabel(r'$\langle X_1 \rangle$')

axs[2, 2].plot(tlist, res_sq.expect[2], label='Single Bath Spin', linestyle='-', color='g')
axs[2, 2].set_ylabel(r'$\langle Y_1 \rangle$')
axs[2, 2].set_xlabel('Time')

for ax in axs.flat:
    ax.legend(loc='best')
    ax.grid(True)

plt.tight_layout()
plt.show()
