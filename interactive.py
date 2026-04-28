import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# FAST interactive QCRB vs Omega
#
# This version avoids SymPy entirely at runtime.
# It uses the closed-form approximate r_nm(t; J) directly and
# computes d rho / dJ numerically with a central finite difference.
#
# QCRB plotted:
#     Delta J_QCRB(Omega) = 1 / sqrt(max_t F_Q(J, t; Omega))
# ============================================================

# -------------------------
# User parameters
# -------------------------
N = 6
J_val = 0.2
r0_val = 1.0
x0_val = 1.0

Omega_min = 0.5
Omega_max = 20.0
num_Omega = 100
Omega_grid = np.linspace(Omega_min, Omega_max, num_Omega)

t_min = 0.0
t_max = 25.0
num_t = 160
tlist = np.linspace(t_min, t_max, num_t)

beta0 = 0.03
gamma0 = 0.02
TOL = 1e-12
FD_EPS = 1e-6

# -------------------------
# Replace these with your real rho0_mat and Jz if available
# -------------------------
m_vals = np.linspace(-(N - 1) / 2, (N - 1) / 2, N)
Jz = np.diag(m_vals)

psi0 = np.ones(N, dtype=np.complex128)
psi0 /= np.linalg.norm(psi0)
rho0_mat = np.outer(psi0, psi0.conj())

m_vals_sol2 = np.diag(Jz)
n_mat, m_mat = np.meshgrid(m_vals_sol2, m_vals_sol2, indexing='ij')

# Precompute integer-like combinations used everywhere
nm_diff = n_mat - m_mat
nm_sum = n_mat + m_mat
nm_sqdiff = n_mat**2 - m_mat**2


def r_nm_closed(J, Omega, gamma, beta, t, r0=1.0, x0=1.0):
    """Closed-form approximate r_nm(t;J) using corrected signs."""
    C1 = (
        2.0 * gamma
        + (J**2 * (beta + gamma) * nm_diff**2) / (2.0 * Omega**2)
        + (J**2 * beta * nm_sum**2) / (2.0 * Omega**2)
    )

    C0 = (
        (J**2 * gamma * (beta + gamma) * nm_diff**2) / (Omega**2)
        + (J**4 * nm_sqdiff**2) / (4.0 * Omega**2)
    )

    Delta = np.sqrt(C1**2 - 4.0 * C0 + 0j)

    numerator = (
        1j * J**2 * nm_sqdiff * x0 / Omega
        - (J**2 * (beta + gamma) * nm_diff**2 * r0) / (Omega**2)
        + C1 * r0
    )

    cosh_term = np.cosh(Delta * t / 2.0)
    sinh_term = np.sinh(Delta * t / 2.0)

    safe_sinh_over_delta = np.empty_like(Delta, dtype=np.complex128)
    mask = np.abs(Delta) > 1e-14
    safe_sinh_over_delta[mask] = sinh_term[mask] / Delta[mask]
    safe_sinh_over_delta[~mask] = t / 2.0

    r_nm_t = np.exp(-C1 * t / 2.0) * (
        r0 * cosh_term + numerator * safe_sinh_over_delta
    )
    return r_nm_t


def rho_of_J(J, Omega, gamma, beta, t):
    r_nm_t = r_nm_closed(J, Omega, gamma, beta, t, r0=r0_val, x0=x0_val)
    rho = rho0_mat * r_nm_t
    rho = 0.5 * (rho + rho.conj().T)

    tr = np.trace(rho)
    if abs(tr) > TOL:
        rho = rho / tr
    return rho


def drho_dJ_fd(J, Omega, gamma, beta, t, eps=FD_EPS):
    rho_p = rho_of_J(J + eps, Omega, gamma, beta, t)
    rho_m = rho_of_J(J - eps, Omega, gamma, beta, t)
    return (rho_p - rho_m) / (2.0 * eps)


def qfi_mixed(rho, drho, tol=TOL):
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)

    evals, evecs = np.linalg.eigh(rho)
    evals = np.real(evals)
    evals[evals < 0.0] = 0.0

    # Transform drho to eigenbasis once
    M = evecs.conj().T @ drho @ evecs
    denom = evals[:, None] + evals[None, :]
    mask = denom > tol

    F = np.sum(2.0 * np.abs(M[mask])**2 / denom[mask])
    return float(np.real(F))


def qfi_vs_time_for_single_omega(Omega, gamma, beta):
    qfi_t = np.empty_like(tlist, dtype=float)
    for i, t in enumerate(tlist):
        rho = rho_of_J(J_val, Omega, gamma, beta, t)
        drho = drho_dJ_fd(J_val, Omega, gamma, beta, t)
        qfi_t[i] = qfi_mixed(rho, drho)
    return qfi_t


def qcrb_vs_omega(gamma, beta):
    qcrb = np.empty_like(Omega_grid, dtype=float)
    t_opt = np.empty_like(Omega_grid, dtype=float)

    for i, Omega in enumerate(Omega_grid):
        qfi_t = qfi_vs_time_for_single_omega(Omega, gamma, beta)
        idx = np.argmax(qfi_t)
        f_best = qfi_t[idx]
        t_opt[i] = tlist[idx]
        qcrb[i] = np.inf if f_best <= TOL else 1.0 / np.sqrt(f_best)

    return qcrb, t_opt


# Initial data
qcrb_init, t_opt_init = qcrb_vs_omega(gamma0, beta0)

# -------------------------
# Plot
# -------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
plt.subplots_adjust(left=0.12, bottom=0.22, hspace=0.28)

(line_qcrb,) = ax1.plot(Omega_grid, qcrb_init, lw=2)
ax1.set_ylabel(r'$\Delta J_{\mathrm{QCRB}} = 1/\sqrt{F_Q^{\max}}$')
ax1.set_title('Optimal-time QCRB vs Omega')
ax1.grid(True, alpha=0.3)

(line_topt,) = ax2.plot(Omega_grid, t_opt_init, lw=2)
ax2.set_xlabel(r'$\Omega$')
ax2.set_ylabel(r'$t_{\mathrm{opt}}$')
ax2.grid(True, alpha=0.3)

status_text = fig.text(
    0.12,
    0.96,
    f'gamma={gamma0:.4f}, beta={beta0:.4f}, J={J_val:.4f}',
    fontsize=10
)

ax_gamma = plt.axes([0.12, 0.10, 0.76, 0.03])
ax_beta = plt.axes([0.12, 0.05, 0.76, 0.03])

slider_gamma = Slider(ax_gamma, 'gamma', 0.0, 0.5, valinit=gamma0, valstep=0.002)
slider_beta = Slider(ax_beta, 'beta', 0.0, 0.5, valinit=beta0, valstep=0.002)


def update(_):
    gamma = slider_gamma.val
    beta = slider_beta.val

    qcrb_new, t_opt_new = qcrb_vs_omega(gamma, beta)

    line_qcrb.set_ydata(qcrb_new)
    line_topt.set_ydata(t_opt_new)

    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    status_text.set_text(f'gamma={gamma:.4f}, beta={beta:.4f}, J={J_val:.4f}')
    fig.canvas.draw_idle()


slider_gamma.on_changed(update)
slider_beta.on_changed(update)

plt.show()
