import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt

matmul_lib = ctypes.CDLL("libmatmul.dylib")

matmul_lib.matmul_naive.argtypes = [
    ctypes.POINTER(ctypes.c_float),  
    ctypes.POINTER(ctypes.c_float),  
    ctypes.POINTER(ctypes.c_float),  
    ctypes.c_int
]
matmul_lib.get_naive_cpu_time.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
matmul_lib.get_naive_cpu_time.restype = ctypes.c_double 

matmul_lib.matmul_optimized.argtypes = matmul_lib.matmul_naive.argtypes
matmul_lib.get_optimized_cpu_time.argtypes = matmul_lib.get_naive_cpu_time.argtypes
matmul_lib.get_optimized_cpu_time.restype = ctypes.c_double  

metal_lib = ctypes.CDLL("libmetal_backend.dylib")

metal_lib.run_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]

metal_lib.get_gpu_time.argtypes = [ctypes.POINTER(ctypes.c_double)]

def naive_matmul(A: np.ndarray, B: np.ndarray):
    size = A.shape[0]
    C = np.zeros((size, size), dtype=np.float32)

    ptrA = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptrB = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptrC = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    matmul_lib.matmul_naive(ptrA, ptrB, ptrC, size)
    return C

def optimized_cpu_matmul(A: np.ndarray, B: np.ndarray):
    size = A.shape[0]
    C = np.zeros((size, size), dtype=np.float32)

    ptrA = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptrB = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptrC = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    matmul_lib.matmul_optimized(ptrA, ptrB, ptrC, size)
    return C

def metal_matmul(A: np.ndarray, B: np.ndarray):
    size = A.shape[0]
    C = np.zeros((size, size), dtype=np.float32)

    ptrA = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptrB = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptrC = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    metal_lib.run_matmul(ptrA, ptrB, ptrC, size)
    return C

sizes = [64, 128, 256, 512, 1024]
num_iterations = 10

naive_times, opt_times, gpu_times = [], [], []
speedup_naive, speedup_opt = [], []

for s in sizes:
    print(f"\nMatrix size: {s}x{s}")

    A = np.random.rand(s, s).astype(np.float32)
    B = np.random.rand(s, s).astype(np.float32)

    naive_total, opt_total, gpu_total = 0, 0, 0

    for _ in range(num_iterations):
        ptrA, ptrB = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ptrC = np.zeros((s, s), dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        naive_total += matmul_lib.get_naive_cpu_time(ptrA, ptrB, ptrC, s) / 1000
        opt_total += matmul_lib.get_optimized_cpu_time(ptrA, ptrB, ptrC, s) / 1000

        gpuTimePtr = ctypes.c_double()
        metal_matmul(A, B)
        metal_lib.get_gpu_time(ctypes.byref(gpuTimePtr))
        gpu_total += gpuTimePtr.value / 1000

    naive_times.append(naive_total / num_iterations)
    opt_times.append(opt_total / num_iterations)
    gpu_times.append(gpu_total / num_iterations)
    speedup_naive.append(naive_times[-1] / gpu_times[-1])
    speedup_opt.append(opt_times[-1] / gpu_times[-1])

plt.plot(sizes, naive_times, label="Naïve CPU")
plt.plot(sizes, opt_times, label="Optimized CPU")
plt.plot(sizes, gpu_times, label="Metal GPU")
plt.legend()
plt.show()
