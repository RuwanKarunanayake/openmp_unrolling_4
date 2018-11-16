import os
import time

number_of_times = 10;

start_time = time.time()
for i in range(0,number_of_times):
	os.system('/home/eranga/MyWork/FinalProject/computeBSSN 0 3 5')
time_for_unimproved_version = (time.time() - start_time)/number_of_times


start_time = time.time()
for i in range(0,number_of_times):
	os.system('./build/computeBSSN 0 3 5')
time_for_improved_version = (time.time() - start_time)/number_of_times

print("speedUp: %s " % (time_for_unimproved_version / time_for_improved_version))