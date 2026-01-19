import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sympy as sp
import math
def cm(N,m):
    J = N//2
    c = math.comb(N, J + m)
    return (1/(2**(J))) * np.sqrt(c)

def s_p(N, m):
    J = N//2
    return np.sqrt((J-m)(J+m+1))
def s_m(N, m):
    J = N//2
    return np.sqrt((J+m)(J-m+1))

def Sy(N, m):
    return 0.5 * 1j * (s_m(N, m) - s_p(N, m))

N = 10
psi_0 = np.array([cm(N, m) for m in range(-N//2, N//2 + 1)])

exp_sy = np.vdot(psi_0, Sy(N, 0) @ psi_0)
print(exp_sy)


