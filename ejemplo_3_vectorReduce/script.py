import os
import time

FILE = "atomic.csv"
COMPILED_FILE = f"./cuda_vectorReduce"

print(f"Compiling {COMPILED_FILE}...")
os.system(f"make")

os.system(f"rm -f {FILE}")
os.system(f"echo  >> {FILE}")

DIM_VEC = [512, 1024, 2048, 4096]
DIM_BLOCK = [1, 2, 4, 8, 16]

os.system(f"echo \"Vector Length;Block Size;Time\" >> {FILE}")


for i in DIM_VEC:
    for j in DIM_BLOCK:
        os.system(f"{COMPILED_FILE} --n={i} --bsx={j} >> {FILE}")