import sympy as sp

# ============================================================
# Symbolic QFI with respect to J using the SLD equation
#
# d rho / dJ = 1/2 (rho L + L rho)
# F_Q = Tr[rho L^2]
#
# No simplification is applied.
# ============================================================

# -------------------------
# Symbols
# -------------------------
J, Omega, gamma, beta, t = sp.symbols('J Omega gamma beta t')
r0, x0 = sp.symbols('r0 x0')
I = sp.I

# Hilbert-space dimension
N = 2   # start with 2; increase only if SymPy can handle it

# symbolic eigenvalues of Jz/Sz basis
m_vals = sp.symbols(f'm0:{N}')

# symbolic initial density matrix rho0
rho0 = sp.Matrix(N, N, lambda a, b: sp.symbols(f'rho0_{a}_{b}', complex=True))

# -------------------------
# Define r_nm(t; J)
# -------------------------
def r_nm_symbolic(n, m):
    C1 = (
        2*gamma
        + J**2*(beta + gamma)*(n - m)**2/(2*Omega**2)
        + J**2*beta*(m + n)**2/(2*Omega**2)
    )

    C0 = (
        J**2*gamma*(beta + gamma)*(n - m)**2/(Omega**2)
        + J**4*(n**2 - m**2)**2/(4*Omega**2)
    )

    Delta = sp.sqrt(C1**2 - 4*C0)

    numerator = (
        I*J**2*(n**2 - m**2)*x0/Omega
        - J**2*(beta + gamma)*(n - m)**2*r0/(Omega**2)
        + C1*r0
    )

    rnm = sp.exp(-C1*t/2) * (
        r0*sp.cosh(Delta*t/2)
        + numerator * sp.sinh(Delta*t/2) / Delta
    )

    return rnm


# -------------------------
# Build symbolic rho(J,t)
# -------------------------
rho = sp.Matrix.zeros(N, N)

for a in range(N):
    for b in range(N):
        n = m_vals[a]
        m = m_vals[b]
        rho[a, b] = rho0[a, b] * r_nm_symbolic(n, m)

# -------------------------
# Symbolic derivative wrt J
# -------------------------
drho_dJ = rho.diff(J)

# -------------------------
# Symbolic SLD matrix L
# -------------------------
L_symbols = sp.symbols(f'L0:{N*N}', complex=True)
L = sp.Matrix(N, N, L_symbols)

# SLD equation:
# rho L + L rho = 2 drho_dJ
SLD_eq_mat = rho*L + L*rho - 2*drho_dJ

eqs = []

for a in range(N):
    for b in range(N):
        eqs.append(sp.Eq(SLD_eq_mat[a, b], 0))

# Solve symbolically for L entries
sol_L = sp.solve(eqs, L_symbols, dict=True, simplify=False)

if len(sol_L) == 0:
    raise RuntimeError("SymPy could not solve the symbolic SLD equations.")

sol_L = sol_L[0]

L_solved = L.subs(sol_L)

# -------------------------
# Symbolic QFI
# -------------------------
FQ_symbolic = sp.Trace(rho * L_solved * L_solved).doit()

print("rho(J,t) =")
sp.print_latex(rho)

print("\nd rho / dJ =")
sp.print_latex(drho_dJ)

print("\nL_J =")
sp.print_latex(L_solved)

print("\nF_Q(J,t) =")
sp.print_latex(FQ_symbolic)