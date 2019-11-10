#!/usr/bin/python3

### Problem is generating the initial contract is inefficient
# either we should do it in Python and store it in memory or we need a different way
import argparse
import numpy as np
import time
import multiprocessing
import sys
#from iolib import ioOutputExcel
from iolib import ioInputJSON
import re
import copy
import random
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
from pycuda import characterize
from pycuda.tools import register_dtype
import scipy.special
import math
import json 
parser = argparse.ArgumentParser(description='Scheduling EV Vehicles to Chargers Through Two-sided Iterative Auction')
parser.add_argument('num_chargers', type=int, help='number of chargers to install')
parser.add_argument('infile', nargs='+', help='json file with route information')
parser.add_argument('--optimal', action='store_true', help='do brute force solution rather than approximation')
args = parser.parse_args()

NUM_CHARGERS = args.num_chargers

NUM_RUNS_PER_BLOCK = 1
NUM_RUNS = 1
NUM_ROUNDS = 100
NUM_PERMUTATIONS = 32

# 2kWh/km
#https://www.sciencedirect.com/science/article/abs/pii/S0360544217301081
#electric bus energy consumption is 1.24-2.48 kWh/km
ENERGY_USED_PER_KM = 2

#https://new.siemens.com/global/en/products/mobility/road-solutions/electromobility/ebus-charging.html
#250kW to 600kW offboard charging  4.1666-10kw = 2-5km charging perminute
#60kW to 120kW onboard
ENERGY_CHARGE_PER_MINUTE = 5
STOP_WAIT_MINUTES = 1

ANNEALING_THRESHOLD_INITIAL = .3
ANNEALING_COOLING_STEP = 0.01
ANNEALING_COOLING_FREQUENCY = int(NUM_ROUNDS/((1.0-ANNEALING_THRESHOLD_INITIAL)/ANNEALING_COOLING_STEP))

with open("kernel.c") as f:
    kernel_approx_src = f.read()
    

# key is the stop lat/lon
# using a dict means we can detect when mutliple routes have the same stop location
# value is a dict with global stop id, and an array of tuples listing route ID and index into the route
stops = {}

ioIn = ioInputJSON()
for f in args.infile:
    ioIn.addFile(f)
    
# an array where each entry is a dict of route data
# that dict contains, among other things, the distance in meters, and an array of stops
routes = ioIn.getRoute()

NUM_ROUTES = len(routes)
LONGEST_ROUTE = 0

for r in routes:
    if len(r) > LONGEST_ROUTE:
        LONGEST_ROUTE = len(r)


preamble = """
struct route_stop_s {
    unsigned int station_id;
    float distance_m;
};
"""
route_stop_dtype = np.dtype([("station_id", np.uint32), ("distance_m", np.float32)])
register_dtype(route_stop_dtype, "route_stop_s")

# make a matrix where each row is a route and the columns are the list of stops. Each stop is two 32 bit values
# the first is the global station_id number and the second is the distance in meters to the next station
# we'll also have an array that lists how many entries are in each row since not all routes have the same number
routes_np = np.empty((NUM_ROUTES, LONGEST_ROUTE), dtype=route_stop_dtype)
routes_lengths_np = np.empty((NUM_ROUTES), dtype=np.uint32)

stopid_to_gps = {}
global_stop_id = 0
for r_index, r in enumerate(routes):
    routes_lengths_np[r_index] = len(r)
    for s_index, s in enumerate(r):
        gps = (s["lat"], s["lon"])
        if gps not in stops:
            stops[gps] = {}
            stops[gps]["routes"] = []
            stops[gps]["global_stop_id"] = global_stop_id
            global_stop_id += 1
        stops[gps]["routes"].append((r_index, s_index))

        routes_np[r_index][s_index]["station_id"] =  stops[gps]["global_stop_id"]
        routes_np[r_index][s_index]["distance_m"] =  s["distance_m"]
print("Route lengths " + str(routes_lengths_np))
NUM_STOPS = global_stop_id

for gps in stops.keys():
    stop_id = stops[gps]["global_stop_id"]
    stopid_to_gps[stop_id] = gps
    
# the stop with the most number of routes crossing it
LONGEST_STOPS = 0
for s in stops.keys():    
    if len(stops[s]["routes"]) > LONGEST_STOPS:
        LONGEST_STOPS = len(stops[s]["routes"])

# this uint32 is packed with the route id in the upper 16 bits and the index into the route in the lower 16
stops_np = np.zeros((NUM_STOPS, LONGEST_STOPS), dtype=np.uint32)
stops_lengths_np = np.empty((NUM_STOPS), dtype = np.uint32)

# convert the stop-> route mapping to a numpy matrix
for s in stops.keys():
    stops_lengths_np[stops[s]["global_stop_id"]] = len(stops[s]["routes"])
    for index, r in enumerate(stops[s]["routes"]):
        (route_id, stop_index) = r
        #print("Saving route/stop " + str(route_id) + " " + str(stop_index) + " for stop id " + str(stops[s]["global_stop_id"]) + " index " + str(index))
        stops_np[stops[s]["global_stop_id"]][index] = (route_id << 16) + stop_index
#import pdb; pdb.set_trace()
#print("XXX  " + str(stops_np))
NUM_CHARGER_INTS = int(NUM_STOPS/32)
if NUM_STOPS % 32 > 0:
    NUM_CHARGER_INTS += 1

# generate one different initial permutation for each run
chargers_np = np.zeros((NUM_RUNS, NUM_CHARGER_INTS), dtype=np.uint32)
for r in range(0, NUM_RUNS):
    charger_np = np.zeros((NUM_CHARGER_INTS), dtype=np.uint32)
    charger_list = random.sample(range(0, NUM_STOPS), NUM_CHARGERS)
    for c in charger_list:
        int_index = int(c/32)
        int_shift = int(c%32)
        charger_np[int_index] |= (1<<int_shift)
    chargers_np[r] = charger_np # this assignment copies the whole array

for i in range(0, NUM_STOPS):
    a = (chargers_np[0][int(i/32)] >> (i%32)) & 0x1
    sys.stdout.write(str(a) + " ")
print("")
# Initialize the RNG
init_rng_src = """
#include <curand_kernel.h>

extern "C"
{

__global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
{
        int id = blockIdx.x*blockDim.x + threadIdx.x;

        if (id >= nthreads)
            return;

        curand_init(seed+id, id, offset, &s[id]);
}

} // extern "C"
"""
rng_states_gpu = cuda.mem_alloc(NUM_RUNS*32*NUM_RUNS_PER_BLOCK*characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
module = SourceModule(init_rng_src, no_extern_c=True)
init_rng = module.get_function('init_rng')
init_rng(np.int32(NUM_RUNS*32*NUM_RUNS_PER_BLOCK), rng_states_gpu, np.uint32(time.time()), np.uint64(0), block=(32,NUM_RUNS_PER_BLOCK,1), grid=(NUM_RUNS,1))
    

is_simiulation = 0
if NUM_RUNS == 1:
    is_simulation = 1
defines = "#define NUM_ROUTES " + str(NUM_ROUTES) + "\n" +\
          "#define NUM_STOPS " + str(NUM_STOPS) + "\n" +\
          "#define NUM_STOPS_INTS " + str(NUM_CHARGER_INTS) + "\n" +\
          "#define SA_THRESHOLD_INITIAL " + str(ANNEALING_THRESHOLD_INITIAL) + "\n" +\
          "#define ROUNDS " + str(NUM_ROUNDS) + "\n" +\
          "#define NUM_CHARGERS " + str(NUM_CHARGERS) + "\n" +\
          "#define ENERGY_USED_PER_KM " + str(ENERGY_USED_PER_KM) + "\n" +\
          "#define ENERGY_CHARGE_PER_MINUTE " + str(ENERGY_CHARGE_PER_MINUTE) + "\n" +\
          "#define STOP_WAIT_MINUTES " + str(STOP_WAIT_MINUTES) + "\n" +\
          "#define ANNEALING_COOLING_FREQUENCY " + str(ANNEALING_COOLING_FREQUENCY) + "\n" +\
          "#define ANNEALING_COOLING_STEP " + str(ANNEALING_COOLING_STEP) + "\n" +\
          "#define LONGEST_ROUTE " + str(LONGEST_ROUTE) + "\n" +\
          "#define LONGEST_STOPS " + str(LONGEST_STOPS) + "\n" +\
          "#define SIMULATION " + str(is_simulation) + "\n"

initial_assignments = None

if args.optimal:
    NUM_THREADS = NUM_RUNS * 32
    TOTAL_WORK = scipy.special.comb(NUM_STOPS, NUM_CHARGERS, exact=True)

    print("Total work items: " + str(TOTAL_WORK))
    # if the total work is more than fits in a 32 bit int then the code will get confused generating the initial assignments
    #assert TOTAL_WORK < 4294967296
    
    # N-1 threads do equal amounts of work. We'll take the floor of that value
    # then the Nth thread does the remainder
    #NUM_WORK_PER_THREAD = int(math.floor(TOTAL_WORK/(NUM_THREADS-1)))
    NUM_WORK_PER_THREAD = int(math.ceil(TOTAL_WORK/(NUM_THREADS)))

    # it is easier to do the same amount of work in the last thread so we back up the
    # offset so that the amount of work is equal. This means we'll repeat some work
    # that other threads did, but it's better than having to special if-case everything
    # due to SIMD semantics
    LAST_THREAD_START_OFFSET = TOTAL_WORK - NUM_WORK_PER_THREAD
    
    defines += "#define NUM_THREADS " + str(NUM_THREADS) + "\n" +\
               "#define TOTAL_WORK " + str(TOTAL_WORK) + "ULL\n" +\
               "#define NUM_WORK_PER_THREAD " + str(NUM_WORK_PER_THREAD) + "\n" +\
               "#define LAST_THREAD_START_OFFSET " + str(LAST_THREAD_START_OFFSET) + "ULL\n"
               
    print("total work %u work per thread %u last thread offset %u"%(TOTAL_WORK, NUM_WORK_PER_THREAD, LAST_THREAD_START_OFFSET))

    """
    initial_assignments = np.zeros((NUM_THREADS, NUM_CHARGERS), dtype=np.uint32)

    work_assigned = 0
    current_assignments = np.zeros((NUM_CHARGERS), dtype=np.uint32)
    for i in range(0, NUM_CHARGERS):
        current_assignments[i] = i

        
    increment_desired = NUM_WORK_PER_THREAD
    # for each thread, generate the offset it starts doing work at
    for i in range(0, NUM_THREADS):
        # save the assignments for this thread
        initial_assignments[i] = current_assignments
        print("XXX saving " + str(current_assignments) + " at thread " + str(i) + " assigned " + str(work_assigned))
        work_assigned += increment_desired
        
        # if this is the last thread, we're done; don't bother incrementing the current_assignments
        # it's actually dangerous to increment it 
        if i == NUM_THREADS-1:
            break

        # if adding NUM_WORK_PER_THREAD will cause that thread to run over when it does its work
        # adjust the amount we increment so that the remaining threads start from a point where
        # doing NUM_WORK_PER_THREAD takes them exactly to the end
        if work_assigned + (2*NUM_WORK_PER_THREAD) > TOTAL_WORK:
            increment_desired = (TOTAL_WORK-NUM_WORK_PER_THREAD) - work_assigned
            
        # increment the current_assignments by increment_desired
        for j in range(0, increment_desired):
            farthest_index = NUM_CHARGERS+1
            for k in range(NUM_CHARGERS-1, -1, -1):
                current_assignments[k] += 1
                farthest_index = k
                if current_assignments[k] < NUM_STOPS- (NUM_CHARGERS - 1 - k):
                    break

            for k in range(farthest_index+1, NUM_CHARGERS):
                current_assignments[k] = current_assignments[k-1]+1
        
    sys.exit()

    """
else:
    defines += "#define NUM_THREADS " + str(999) + "\n" +\
               "#define TOTAL_WORK " + str(0) + "\n" +\
               "#define NUM_WORK_PER_THREAD " + str(0) + "\n" +\
               "#define LAST_THREAD_START_OFFSET " + str(0) + "\n"
    
mod = SourceModule(preamble + defines + kernel_approx_src, no_extern_c=True)

(routes_lengths_gpu, size_in_bytes) = mod.get_global("routes_lengths")
(stops_lengths_gpu, size_in_bytes) = mod.get_global("stops_lengths")

cuda.memcpy_htod(routes_lengths_gpu, routes_lengths_np)
cuda.memcpy_htod(stops_lengths_gpu, stops_lengths_np)

routes_gpu = cuda.mem_alloc(routes_np.nbytes)
cuda.memcpy_htod(routes_gpu, routes_np)

stops_gpu = cuda.mem_alloc(stops_np.nbytes)
cuda.memcpy_htod(stops_gpu, stops_np)

final_utilities_np = np.zeros((NUM_RUNS),dtype=np.float32)
final_utilities_gpu = cuda.mem_alloc(final_utilities_np.nbytes)

final_chargers_np = np.zeros((NUM_RUNS, NUM_CHARGER_INTS), dtype=np.uint32)
final_chargers_gpu = cuda.mem_alloc(final_chargers_np.nbytes)
cuda.memcpy_htod(final_chargers_gpu, chargers_np)

simulation_data_np = np.zeros((NUM_ROUNDS, NUM_CHARGER_INTS),dtype=np.uint32)
simulation_data_gpu = cuda.mem_alloc(simulation_data_np.nbytes)

if args.optimal:
    func = mod.get_function("run_brute_force")
    func(routes_gpu, stops_gpu, final_utilities_gpu, final_chargers_gpu, block=(32,NUM_RUNS_PER_BLOCK,1), grid=(NUM_RUNS,1))

else:
    func = mod.get_function("run_approximation")
    func(routes_gpu, stops_gpu, rng_states_gpu, final_utilities_gpu, final_chargers_gpu, simulation_data_gpu, block=(32,NUM_RUNS_PER_BLOCK,1), grid=(NUM_RUNS,1))


cuda.memcpy_dtoh(final_utilities_np, final_utilities_gpu)

cuda.memcpy_dtoh(final_chargers_np, final_chargers_gpu)

cuda.memcpy_dtoh(simulation_data_np, simulation_data_gpu)

max_utility = 0.0
max_utility_i = -1

for utility_i, utility in enumerate(final_utilities_np):
    if utility > max_utility:
        max_utility = utility
        max_utility_i = utility_i


chargers_list = {}
for i in range(0, NUM_STOPS):
    if ((final_chargers_np[max_utility_i][int(i/32)] >> (i%32)) & 0x1) == 1:
        sys.stdout.write(str(i) + "  ")
        chargers_list[i] = 1
sys.stdout.write("  with utility " + str(final_utilities_np[r]))
print("")
print("Routes:")
for index, r in enumerate(routes_np):
    for i in range(0, routes_lengths_np[index]):
        sys.stdout.write(str(routes_np[index][i]["station_id"]))
        if routes_np[index][i]["station_id"] in chargers_list:
            sys.stdout.write("*")
        sys.stdout.write("  ")
    print("\n")

if args.optimal:
    print("Runs: %d" % (len(final_utilities_np)))
    print("Utility Max: %f" % (max(final_utilities_np)))
else:
    
    print("Runs: %d" % (len(final_utilities_np)))
    print("Rounds: %d" % (NUM_ROUNDS))
    print("Utility Max: %f" % (max(final_utilities_np)))
    print("Utility Min: %f" % (min(final_utilities_np)))
    print("Utility Avg: %f" % (np.mean(final_utilities_np)))
    print("Utility Median: %f" % (np.median(final_utilities_np)))
    print("Utility Stdev: %f" % (np.std(final_utilities_np)))


    
if NUM_RUNS == 1:
    for i in range(0, NUM_STOPS):
        temp = {}
        temp["action_time"] = 0
        temp["coordinates"] = []
        temp["station_id"] = i
        temp["coordinates"].append(stopid_to_gps[i])
        temp["type"] = "station_create"
        print(json.dumps(temp, sort_keys=True))
        
    chargers = {}
    for round in range(0, NUM_ROUNDS):
        for i in range(0, NUM_STOPS):
            bit_uint = int(i/32)
            bit_offset = i%32
            bit_value = (simulation_data_np[round][bit_uint] >> bit_offset) & 0x1

            temp = {}
            temp["action_time"] = round
            temp["coordinates"] = []
            temp["coordinates"].append(stopid_to_gps[i])
            temp["station_id"] = i
            
            # if the charger is not here but it was last round, we removed it
            if bit_value == 0 and i in chargers:
                temp["type"] = "station_off"
                print(json.dumps(temp, sort_keys=True))
                del chargers[i]
            # if the charger is here but it wasn't last round, we added it
            elif bit_value == 1 and i not in chargers:
                temp["type"] = "station_on"
                print(json.dumps(temp, sort_keys=True))
                chargers[i] = 1
            # other cases mean the station stayed the same so we don't update


