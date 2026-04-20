import traceback
import sys
try:
    import qutip
    print("qutip imported successfully, version:", qutip.__version__)
    import numpy
    print("numpy version:", numpy.__version__)
    import scipy
    print("scipy version:", scipy.__version__)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
