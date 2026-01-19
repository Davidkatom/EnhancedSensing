
import numpy as np
import scipy.linalg
import sympy as sp
import math

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

# The user's code constructs a diagonal matrix, which is physically wrong for Sy in z-basis
Sy_matrix = np.diag([Sy(N, m) for m in range(-N//2, N//2 + 1)])
Sy_2_matrix = Sy_matrix @ Sy_matrix

psi_0 = np.array([cm(N, m) for m in range(-N//2, N//2 + 1)])

exp_sy = np.vdot(psi_0, Sy_matrix @ psi_0)  
exp_sy_2 = np.vdot(psi_0, Sy_2_matrix @ psi_0)
print(f"Numerical <Sy> (broken): {exp_sy}")
print(f"Numerical <Sy^2> (broken): {exp_sy_2}")
