import os
import time

FILE = "file-1.csv"
COMPILED_FILE = "./compara_kernels"

os.system("rm -f " + COMPILED_FILE)
os.system("make")


os.system("echo  >> " + FILE)

DIM_MAT = [512, 1024, 2048, 4096]
TAM_BLO = [4, 8, 16, 32]
KERNELS = [1, 2, 3]

os.system("echo \"Kernel;Dim mat;Block Size;Time (ms)\" >> " + FILE)

for k in KERNELS:
    for i in DIM_MAT:
        for b in TAM_BLO:
            os.system(COMPILED_FILE + " --N=" + str(i) + " --W=" +  str(b) + " --K=" + str(k) + " >> " + FILE)