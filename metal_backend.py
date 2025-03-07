import ctypes
import numpy as np
import time

metal_lib = ctypes.CDLL("libmetal_backend.dylib")

metal_lib.run_matmul.argtypes = [
  ctypes.POINTER(ctypes.c_float), #A
  ctypes.POINTER(ctypes.c_float), #B
  ctypes.POINTER(ctypes.c_float), #C
  ctypes.c_int #size
]

def metal_matmul(A :np.ndarray, B: np.ndarray):
  size = A.shape[0]
  C = np.zeros((size, size), dtype=np.float32)
  
  ptrA = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  ptrB = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  ptrC = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

  metal_lib.run_matmul(ptrA, ptrB, ptrC, size)
  return C

sizes = [64, 128, 256, 512, 1024]

for s in sizes:
  print(f"matrix size: {s}x{s}\n")
  A = np.random.rand(s, s).astype(np.float32)
  B = np.random.rand(s, s).astype(np.float32)
  
  if (A @ B == metal_matmul(A, B)).all():
     print("metal code and numpy return same things")
  else:
      print("your metal code is messed up")

  print("Matrix A:")
  print(A)

  print("\nMatrix B:")
  print(B)

  print("\nNumPy Result (A @ B):")
  print(A @ B)

  print("\nMetal GPU Result:")
  print(metal_matmul(A, B))

  start = time.time()
  cpuComp = A @ B
  cpuTime = time.time() - start 
  print(f"cpu time: {cpuTime} sec")
  
  start = time.time()
  c_metal = metal_matmul(A, B)
  gpuTime = time.time() - start
  print(f"gpu time: {gpuTime} sec")
  
  speedupFactor = cpuTime /gpuTime
  print(f"speedup factor: {speedupFactor}")