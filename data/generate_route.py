#!/usr/bin/python3

import argparse
import json
import requests
import time
import copy
import sys

parser = argparse.ArgumentParser(description='convert algorithm output into simulation format')
parser.add_argument('infile', help='json file with the input data')
args = parser.parse_args()



fh = open(args.infile, "r")
route = []
for line in fh:
    if len(line) < 5 or line[0] == '#':
        continue
    route.append(line)
fh.close()

# append the first station to the end so we calculate the distance from the last station back to the first
route.append(route[0])

start_lat = None
start_lon = None
total_distance = 0
station_id = 0
route_data = {}
route_data["stop_list"] = []

for line in route:
    coords = line.strip().split(',')
    assert len(coords) == 2
    lat = coords[0]
    lon = coords[1]

    # first row, no distance yet
    if start_lat is None:
        start_lat = lat
        start_lon = lon
        continue
    
    req_str = "http://localhost:8989/route?point=" + str(start_lat) + "," + str(start_lon) + "&point=" + str(lat) + "," + str(lon) + "&vehicle=car&points_encoded=false"
    call = requests.get(req_str)
    #print(call.status_code, call.reason)
    #print(call.text)
    assert call.status_code==200
    api_data = json.loads(call.text)
    
    #processed_data["coordinates"] = api_data["paths"][0]["points"]["coordinates"]
    distance_m = api_data["paths"][0]["distance"]
    total_distance += distance_m

    stop_data = {"index" : station_id,
                 "lat" : start_lat,
                 "lon" : start_lon,
                 "distance_m" : distance_m}

    route_data["stop_list"].append(stop_data)
    
    start_lat = lat
    start_lon = lon

    station_id+=1

route_data["meters"] = total_distance

print(json.dumps(route_data, indent=4, sort_keys=True))

