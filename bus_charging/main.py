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

from pycuda.curandom import rand as curand
from pycuda import characterize
from pycuda.tools import register_dtype

parser = argparse.ArgumentParser(description='Scheduling EV Vehicles to Chargers Through Two-sided Iterative Auction')
parser.add_argument('infile', nargs='+', help='json file with route information')

args = parser.parse_args()

# key is the stop lat/lon
# value is a hash with global stop id, and an array of tuples listing route ID and index into the route
stops = {}

ioIn = ioInputJSON(args.infile[0])

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
    print(" r is " + str(r))
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


# the stop with the most number of routes crossing it
LONGEST_STOPS = 0
for s in stops:
    if len(s["routes"]) > LONGEST_STOPS:
        LONGEST_STOPS = len(s["routes"])

# this uint32 is packed with the route id in the upper 16 bits and the index into the route in the lower 16
stops_np = np.empty((global_stop_id, LONGEST_STOPS), dtype=np.uint32)
stops_lengths_np = np.empty((global_stop_id), dtype = np.uint32)

for s in stops:
    stops_lengths_np[s["station_id"]] = len(s["routes"])
    for index, r in enumerate(s["routes"]):
        (route_id, stop_index) = r
        stops_np[s["station_id"]][index] = route_id << 16 + stop_index

#routes_lengths_np and stops_lengths_np get sent into constant cache
# routes_np and stops_np get set as regular global memry

# what's left is to make a list of stops that have a charging station. This is probably a bit array we'll copy into shared memory
# I guess we'll initialize each run with a random arrangement of stations for however many we're installing at this round

