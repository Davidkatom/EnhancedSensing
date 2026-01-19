import numpy as np
import scipy.linalg
import sympy as sp
import math
import matplotlib.pyplot as plt
def cm(N,m):
    J = N//2
    c = math.comb(N, J + m)
    return (1/(2**(J))) * np.sqrt(c)

def s_p(N, m):
    J = N//2
    return np.sqrt((J-m)*(J+m+1))
def s_m(N, m):
    J = N//2
    return np.sqrt((J+m)*(J-m+1))

def Sy(N, m):
    return 0.5 * 1j * (s_m(N, m) - s_p(N, m))

N = 10
dim = N + 1
Sy_matrix = np.zeros((dim, dim), dtype=complex)
for i in range(dim):
    m = i - N//2
    if i + 1 < dim:
        Sy_matrix[i+1, i] = -0.5j * s_p(N, m)
    
    if i - 1 >= 0:
        Sy_matrix[i-1, i] = 0.5j * s_m(N, m)


Sy_2_matrix = Sy_matrix @ Sy_matrix

psi_0 = np.array([cm(N, m) for m in range(-N//2, N//2 + 1)])

exp_sy = np.vdot(psi_0, Sy_matrix @ psi_0)  
exp_sy_2 = np.vdot(psi_0, Sy_2_matrix @ psi_0)
print(f"Musical <Sy>: {np.real_if_close(exp_sy)}")
print(f"Musical <Sy^2>: {np.real_if_close(exp_sy_2)}")

# --- Symbolic ---

m_sym = sp.symbols('m', integer=True)
N_sym = sp.symbols('N', integer=True, positive=True)
g_sym = sp.symbols('g', real=True)
J_sym = sp.Rational(1,2)*N_sym

def cm_sym(N, m, t=0):
    return (1 / 2**(N/2)) * sp.sqrt(sp.binomial(N, N/2 + m).rewrite(sp.factorial)) * sp.exp(-t * m * m * g_sym)

def s_p_sym(N, m):
    J = N/2
    return sp.sqrt((J-m)*(J+m+1))

def s_m_sym(N, m):
    J = N/2
    return sp.sqrt((J+m)*(J-m+1))

term1 = cm_sym(N_sym, m_sym - 1) * cm_sym(N_sym, m_sym) * s_p_sym(N_sym, m_sym - 1)
term2 = cm_sym(N_sym, m_sym + 1) * cm_sym(N_sym, m_sym) * s_m_sym(N_sym, m_sym + 1)

sum_sy = 0.5 * sp.I * sp.Sum(term1 - term2, (m_sym, -J_sym, J_sym))


term1 = cm_sym(N_sym, m_sym + 2) * s_m_sym(N_sym, m_sym + 2) * s_m_sym(N_sym, m_sym + 1)
term2 = cm_sym(N_sym, m_sym - 2) * s_p_sym(N_sym, m_sym - 2) * s_p_sym(N_sym, m_sym - 1)
term3_1 = cm_sym(N_sym, m_sym) * s_m_sym(N_sym, m_sym) * s_p_sym(N_sym, m_sym - 1)
term3_2 = cm_sym(N_sym, m_sym) * s_p_sym(N_sym, m_sym) * s_m_sym(N_sym, m_sym + 1)
term3 = - term3_1 - term3_2

term = (term1 + term2 + term3) * cm_sym(N_sym, m_sym)
exp_sy_2_sym = - 0.25 * sp.Sum(
    term,
    (m_sym, -J_sym, J_sym)
)


# Simplify before printing
sum_sy_simplified = sum_sy.doit()

print("-" * 20)
print("Results:")
print(f"Numerical <Sy>: {np.real_if_close(exp_sy)}")
print(f"Numerical <Sy^2>: {np.real_if_close(exp_sy_2)}")
print("-" * 20)

latex_sy = sp.latex(sum_sy_simplified)
latex_sy2 = sp.latex(exp_sy_2_sym)
latex_sy_n10 = sp.latex(sum_sy.subs(N_sym, 10).doit())
latex_sy2_n10 = sp.latex(exp_sy_2_sym.subs(N_sym, 10).doit())
latex_sy2_term = sp.latex(term.doit().simplify())

print("Symbolic Results (LaTeX):")
print(f"<Sy>: {latex_sy}")
print(f"<Sy^2>: {latex_sy2}")
print("-" * 20)
print("Symbolic Results (N=10) (LaTeX):")
print(f"<Sy> (N=10): {latex_sy_n10}")
print(f"<Sy^2> (N=10): {latex_sy2_n10}")

# Visualization
try:
    import matplotlib.pyplot as plt
    
    # Create a figure to display the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    text_str = (
        r"$\bf{Analytical\ Results}$" + "\n\n" +
        r"$\langle S_y \rangle = " + latex_sy + r"$" + "\n\n" +
        r"$\langle S_y^2 \rangle = " + latex_sy2 + r"$" + "\n\n" +
        r"$\bf{N=10\ Specific\ Case}$" + "\n\n" +
        r"$\langle S_y \rangle_{N=10} = " + latex_sy_n10 + r"$" + "\n\n" +
        r"$\langle S_y^2 \rangle_{N=10} = " + latex_sy2_n10 + r"$" + "\n\n" +
        r"$\bf{Term}$" + "\n\n" +
        r"$" + latex_sy2_term + r"$"
    )
    
    # We use a text box. Matplotlib's mathtext parser supports a subset of TeX.
    # Note: Complex LaTeX from SymPy might exceed mathtext capabilities, 
    # but for simple polynomials/sums it often works.
    
    ax.text(0.1, 0.9, text_str, fontsize=12, verticalalignment='top')
    
    plt.title("Symbolic Calculation Results")
    plt.tight_layout()
    plt.show()
    print("Displayed results in a matplotlib window.")
except ImportError:
    print("Matplotlib not found. Install it to see rendered LaTeX.")
except Exception as e:
    print(f"Could not render LaTeX: {e}")

