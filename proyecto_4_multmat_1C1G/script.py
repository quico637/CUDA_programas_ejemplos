import os
import time

FILE = "file-4.csv"
COMPILED_FILE = "./multmat_1C1G"


os.system("rm -f " + COMPILED_FILE)
os.system("make")

os.system("echo  > \"\"" + FILE)

FILAS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 64]

os.system("echo \"Rows CPU;Time (ms)\" >> " + FILE)

for f in FILAS:
    os.system(COMPILED_FILE + " --M=2048 --N=2048 --K=2048 --W=4 --F=" + str(f) + " >> " + FILE)