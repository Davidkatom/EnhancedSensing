import sys
import traceback

try:
    import numpy
    print("Numpy imported successfully")
    print(f"Version: {numpy.__version__}")
except Exception:
    traceback.print_exc()
