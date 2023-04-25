import os
import time

FILE = "file-2.csv"
COMPILED_FILE = "./mulmatcua_1G"


os.system("rm -f " + COMPILED_FILE)
os.system("make")

os.system("echo  > \"\"" + FILE)

DIM_MAT = [512, 1024, 2048, 4096]
TAM_BLO = [4, 8, 16, 32]

os.system("echo \"Dim mat;Block Size;Time (ms)\" >> " + FILE)


for i in DIM_MAT:
    for b in TAM_BLO:
        os.system(COMPILED_FILE + " --N=" + str(i) + " --W=" +  str(b) + " >> " + FILE)