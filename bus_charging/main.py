#!/usr/bin/python3

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

parser = argparse.ArgumentParser(description='Scheduling EV Vehicles to Chargers Through Two-sided Iterative Auction')
parser.add_argument('num_chargers', type=int, help='number of chargers to install')
parser.add_argument('infile', nargs='+', help='json file with route information')

args = parser.parse_args()

NUM_CHARGERS = args.num_chargers

NUM_RUNS_PER_BLOCK = 1
NUM_RUNS = 1000
NUM_ROUNDS = 1000
NUM_PERMUTATIONS = 32

ENERGY_USED_PER_KM = 2
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
print("XXX route lengths " + str(routes_lengths_np))
NUM_STOPS = global_stop_id

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
          "#define LONGEST_STOPS " + str(LONGEST_STOPS) + "\n"
          
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

func = mod.get_function("run_approximation")

func(routes_gpu, stops_gpu, rng_states_gpu, final_utilities_gpu, final_chargers_gpu, block=(32,NUM_RUNS_PER_BLOCK,1), grid=(NUM_RUNS,1))


cuda.memcpy_dtoh(final_utilities_np, final_utilities_gpu)

cuda.memcpy_dtoh(final_chargers_np, final_chargers_gpu)

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

print("Runs: %d" % (len(final_utilities_np)))
print("Rounds: %d" % (NUM_ROUNDS))
print("Utility Max: %f" % (max(final_utilities_np)))
print("Utility Min: %f" % (min(final_utilities_np)))
print("Utility Avg: %f" % (np.mean(final_utilities_np)))
print("Utility Median: %f" % (np.median(final_utilities_np)))
print("Utility Stdev: %f" % (np.std(final_utilities_np)))
