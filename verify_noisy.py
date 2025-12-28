import sys
import os

# Add relevant paths
sys.path.append(os.path.join(os.getcwd(), 'BruteForceAproach'))

try:
    from fisher_optimize_lambda import optimize_fisher
except ImportError:
    # Try importing directly if running from inside folder
    sys.path.append(os.getcwd())
    from fisher_optimize_lambda import optimize_fisher

print("Running noisy Fisher optimization...")
optimize_fisher(J1=1.0, J2=1.0, gammas=[0.1, 0.1, 0.1])
