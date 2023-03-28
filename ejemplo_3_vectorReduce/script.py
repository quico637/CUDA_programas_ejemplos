import os
import time

FILE = "atomic.csv"
COMPILED_FILE = "./cuda_vectorReduce"

# print("Compiling {COMPILED_FILE}...")
os.system("make")

os.system("rm -f " + FILE)
os.system("echo  >> " + FILE)

DIM_VEC = [512, 1024, 2048, 4096]
DIM_BLOCK = [1, 2, 4, 8, 16]

os.system("echo \"Vector Length;Block Size;Time\" >> " + FILE)


for i in DIM_VEC:
    for j in DIM_BLOCK:
        os.system(COMPILED_FILE + " --n=" + str(i) + " --bsx=" + str(j) + " >> " + FILE)