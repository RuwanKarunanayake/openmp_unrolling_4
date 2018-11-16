import os
import subprocess
import numpy as np

# GPU\CPU Performance test
# change test_param.h accordingly.
# test 1
# isGPU\isCPU 1

runtime = []
runtime_main = []

def print_report():
    runtimeNP = np.array(runtime)
    runtime_avg = np.average(runtimeNP)

    runtime_mainNP = np.array(runtime_main)
    runtime_main_avg = np.average(runtime_mainNP)
    
    print("Speed up: \t=%f "%(runtime_main_avg/runtime_avg))
    print("\n")


os.chdir("../build") # Change to build folder

try:
    os.remove("performance.txt")
except:
    print("File was already deleted")

os.system("cmake ..")
os.system("make all")

blocks = 5

for j in range(5):
    runtime = []
    runtime_main = []
    for i in range(10):
        output = str(subprocess.check_output("./computeBSSN " + str(j) + " " +  str(j) + " " + str(blocks), shell=True))
        output = output.split("\\n")
        run = float(output[-5][20:].strip())
        runtime.append(run)

    for i in range(10):
        # set main thing set previous one here ex: FYP/SymPyGR/bssn/helper/computeBSSN with a space
        output = str(subprocess.check_output("/home/eranga/MyWork/FinalProject/FYPClone/SymPyGR/bssn/build/computeBSSN " + str(j) + " " +  str(j) + " " + str(blocks), shell=True))
        output = output.split("\\n")
        run = float(output[-5][20:].strip())
        runtime_main.append(run)

    print("\nPrint final report")
    print_report()
