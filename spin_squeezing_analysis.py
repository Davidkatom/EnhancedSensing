
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, kron, identity

def main():
    # Parameters
    N = 10  # Reduced to 10 for computational basis (2^11 = 2048 states)
    J = 10.0
    Omega = 10.0
    omega = 1.0
    
    # Time parameters
    t_max = 2.0
    steps = 200
    times = np.linspace(0, t_max, steps)

    print(f"Generating operators for N={N} (Dimension {2**(N+1)})...")

    # Basic Pauli Matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)

    # Helper to construct product operators
    def tensor_list(op_list):
        from functools import reduce
        return reduce(np.kron, op_list)

    # 1. System Operators (S acts on index 0)
    # S_alpha = sigma_alpha (x) I (x) ... (x) I
    
    # Using sparse kron is much better for memory, even at N=10? 
    # 2048x2048 dense is 32MB complex128, totally fine.
    # Let's keep dense for Simplicity with scipy.linalg.eigh
    
    # Construct S_x, S_z
    S_x = np.kron(sx, np.eye(2**N, dtype=complex))
    S_z = np.kron(sz, np.eye(2**N, dtype=complex))
    
    # 2. Bath Operators
    # I_alpha = sum_k (1/2) sigma_alpha^k
    # k runs from 1 to N. System is index 0.
    
    I_x = np.zeros((2**(N+1), 2**(N+1)), dtype=complex)
    I_y = np.zeros((2**(N+1), 2**(N+1)), dtype=complex)
    I_z = np.zeros((2**(N+1), 2**(N+1)), dtype=complex)
    
    # Pre-generate identity lists to avoid re-allocating
    # We need to construct terms like I x I x sigma x I ...
    
    # Full dimension D = 2^(N+1)
    # Efficient construction:
    # Term k is Id_S (x) Id_bath_before (x) sigma (x) Id_bath_after
    
    for k in range(N):
        # k=0 is the first bath spin (index 1 globally)
        # dim_before = 2^(1 + k)  (1 for System + k bath spins)
        # dim_after = 2^(N - 1 - k)
        
        dim_before = 2**(1 + k)
        dim_after = 2**(N - 1 - k)
        
        op_x = np.kron(np.eye(dim_before), np.kron(sx, np.eye(dim_after)))
        op_y = np.kron(np.eye(dim_before), np.kron(sy, np.eye(dim_after)))
        op_z = np.kron(np.eye(dim_before), np.kron(sz, np.eye(dim_after)))
        
        I_x += 0.5 * op_x
        I_y += 0.5 * op_y
        I_z += 0.5 * op_z

    # Hamiltonian
    # H = (Omega/2) sigma_x^S + omega I_x + J sigma_z^S I_z 
    # Note: usually J term is sum J S_z s_z^k = 2 J S_z I_z.
    # The user prompt said: sum (J/2) sigma_z^S sigma_z^k.
    # sum (J/2) sigma_z^S sigma_z^k = (J/2) sigma_z^S (sum sigma_z^k) 
    # = (J/2) sigma_z^S (2 I_z) = J sigma_z^S I_z.
    # So yes, J * S_z * I_z is correct if we assume J defined as in prompt.
    # Actually prompt says: sum (J/2) sigma_z^S sigma_z^k.
    # My code: J * S_z * I_z = J * (sigma_z x I) * I_z
    # I_z = sum (1/2) sigma_z^k
    # So J * S_z * I_z = J * sigma_z^S * sum (1/2) sigma_z^k = sum (J/2) sigma_z^S sigma_z^k.
    # Matches.
    
    # S_x is sigma_x on system. The term is (Omega/2) sigma_x^S.
    # My S_x variable is sigma_x on system. So (Omega/2) * S_x.
    
    Iz2 = I_z @ I_z
    H = (Omega / 2.0) * S_x + omega * I_x + J * (S_z @ Iz2)    
    
    # Initial State
    # |+>_S x |+>...|+> (all N spins)
    # |+> = [1, 1]/sqrt(2)
    plus = np.array([1.0, 1.0]) / np.sqrt(2.0)
    
    # Full state is tensor product of N+1 |+> states
    # Can construct iteratively
    psi_0 = plus
    for _ in range(N):
        psi_0 = np.kron(psi_0, plus)
        
    print("Diagonalizing Hamiltonian...")
    energies, eigenstates = scipy.linalg.eigh(H)
    
    coeffs = eigenstates.T.conj() @ psi_0
    
    sx_variances = []
    sy_variances = []
    sz_variances = []
    
    print("Starting time evolution...")
    
    # Storage for operators in diagonal basis to speed up?
    # No, N=10 is small enough to do vdot every step (~0.1s per step)
    
    for t_idx, t in enumerate(times):
        psi_t = eigenstates @ (coeffs * np.exp(-1j * energies * t))
        
        def expect(Op):
            return np.real(np.vdot(psi_t, Op @ psi_t))
        
        # Means
        mean_x = expect(I_x)
        mean_y = expect(I_y)
        mean_z = expect(I_z)
        
        # Second moments
        mean_x2 = expect(I_x @ I_x)
        mean_y2 = expect(I_y @ I_y)
        mean_z2 = expect(I_z @ I_z)
        
        var_x = mean_x2 - mean_x**2
        var_y = mean_y2 - mean_y**2
        var_z = mean_z2 - mean_z**2
        
        # Normalize by 4/N? 
        # Standard quantum limit for N spins (spin-coherent state) is <J_z^2> = J/2 = N/4.
        # So Var(J_z) / (N/4) = 1.
        # So multiply by 4/N to normalize to 1 at t=0.
        norm = 4.0 / N
        
        sx_variances.append(var_x * norm)
        sy_variances.append(var_y * norm)
        sz_variances.append(var_z * norm)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    ax1.plot(times, sx_variances, label='Var(Sx)')
    ax1.set_ylabel('Norm Var(Sx)')
    ax1.set_title(f'Variance of Sx vs Time (N={N})')
    ax1.axhline(1.0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True)
    # ax1.set_ylim(0.5, 1.5)

    ax2.plot(times, sy_variances, label='Var(Sy)')
    ax2.set_ylabel('Norm Var(Sy)')
    ax2.set_title(f'Variance of Sy vs Time (N={N})')
    ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True)
    # ax2.set_ylim(0.5, 1.5)
    
    ax3.plot(times, sz_variances, label='Var(Sz)')
    ax3.set_ylabel('Norm Var(Sz)')
    ax3.set_title(f'Variance of Sz vs Time (N={N})')
    ax3.set_xlabel('Time')
    ax3.axhline(1.0, color='r', linestyle='--', alpha=0.5)
    ax3.grid(True)
    # ax3.set_ylim(0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('spin_squeezing.png')
    print("Analysis complete. Saved plot to 'spin_squeezing.png'")

    # Find time of maximum squeezing (minimum normalized variance in any direction)
    # We track min var across all components? Or just Y (since X is 0 initially)?
    # Let's find index where min(var_x, var_y, var_z) is minimal (and < 1 ideally).
    # Since var_x starts at 0, that's trivial.
    # We want squeezing *generated* by dynamics, usually in Y/Z plane relative to rotating frame.
    # Let's pick the time where Var(Sy) is minimized (if it goes below 1).
    # Or simply: pick the last time point for demonstration, or a specific interesting time.
    # Let's pick index of minimum Sy variance after t=0.1 (to avoid initial transient if any).
    
    start_idx = int(steps * 0.1)
    # If sy_variances has squeezed values
    min_sy = np.min(sy_variances[start_idx:])
    min_sy_idx = np.argmin(sy_variances[start_idx:]) + start_idx
    
    target_idx = min_sy_idx
    target_time = times[target_idx]
    print(f"Plotting Bloch sphere at t={target_time:.3f} (Min Var(Sy)={min_sy:.3f})")
    
    # Re-compute state at target_time
    psi_target = eigenstates @ (coeffs * np.exp(-1j * energies * target_time))
    
    def expect_target(Op):
        return np.real(np.vdot(psi_target, Op @ psi_target))

    # Mean Spin Vector <I> (normalized to N/2 = 1 for plotting on unit sphere)
    # The bath has N spins. Max spin length is N/2.
    J_vec = np.array([expect_target(I_x), expect_target(I_y), expect_target(I_z)])
    
    # Covariance Matrix Gamma
    # G_ij = 0.5 < {Ii, Ij} > - <Ii><Ij>
    Gamma = np.zeros((3, 3))
    ops = [I_x, I_y, I_z]
    
    for i in range(3):
        for j in range(3):
            # 0.5 <Ii Ij + Ij Ii> = Re <Ii Ij>
            term = np.real(np.vdot(psi_target, ops[i] @ (ops[j] @ psi_target)))
            Gamma[i, j] = term - J_vec[i] * J_vec[j]
            
    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Unit Sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.1)
    
    # Wireframe for better visibility
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1, rstride=10, cstride=10)
    
    # Plot Mean Spin (Normalized)
    # We want to plot it on the sphere.
    # Length of J_vec is approx N/2 for coherent state.
    # Normalize J_vec to 1 for direction.
    J_norm = np.linalg.norm(J_vec)
    J_dir = J_vec / J_norm
    
    # Plot arrow
    ax.quiver(0, 0, 0, J_dir[0], J_dir[1], J_dir[2], length=1.0, color='r', linewidth=2, label='Mean Spin')
    
    # Plot Uncertainty Ellipse
    # The ellipse represents the covariance matrix projected onto the plane perpendicular to J_vec.
    # Actually, the 3D covariance ellipsoid is defined by x^T Gamma^{-1} x = const?
    # Usually we plot the ellipse of standard deviation.
    # Center at J_dir (on the sphere).
    
    # 1. Project Gamma perp to J_dir
    P = np.eye(3) - np.outer(J_dir, J_dir)
    Gamma_perp = P @ Gamma @ P
    
    # 2. Eigenvalues/vectors of projected Gamma
    evals, evecs = np.linalg.eigh(Gamma_perp)
    # Sort eigenvalues
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # The smallest eigenvalue should be 0 (along J_dir direction due to projection).
    # The other two are the variances in the perpendicular plane.
    # We define the semi-axes lengths as sqrt(variance) * scaling_factor
    # Scaling: we visualized J normalized to 1. J_real ~ N/2.
    # Uncertainty relative to J_real is sqrt(Var)/ (N/2).
    # So length on plot = sqrt(Var) / (N/2).
    
    scale = 1.0 / (N/2.0)
    # Magnify for visibility? Let's use exact scale first.
    # Typically 1/sqrt(N) noise. For N=10, 1/3.
    
    # Basis vectors for ellipse
    v1 = evecs[:, 1]
    v2 = evecs[:, 2]
    sig1 = np.sqrt(evals[1]) * scale
    sig2 = np.sqrt(evals[2]) * scale
    
    # Generate ellipse points
    t_circ = np.linspace(0, 2*np.pi, 100)
    # Circle in v1-v2 plane
    # Point = Center + (sig1 cos t) v1 + (sig2 sin t) v2
    ell_points = np.zeros((100, 3))
    center = J_dir
    for i in range(100):
        ell_points[i] = center + sig1 * np.cos(t_circ[i]) * v1 + sig2 * np.sin(t_circ[i]) * v2
        
    ax.plot(ell_points[:, 0], ell_points[:, 1], ell_points[:, 2], color='b', linewidth=2, label='Uncertainty')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Spin State at t={target_time:.3f}')
    ax.legend()
    
    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Equal aspect ratio hack
    # ax.set_aspect('equal') # Not supported in all matplotlib versions for 3D
    
    plt.savefig('bloch_sphere.png')
    print("Saved Bloch sphere plot to 'bloch_sphere.png'")

if __name__ == '__main__':
    main()
