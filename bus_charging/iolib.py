#!/usr/bin/python3

import openpyxl
import statistics
import datetime
import time

import json
import copy
class ioInput:
    def __init__(self):
        self._json_data = None
        self._route = []
        self._meters = None
        
    def getRoute(self):
        return self._route

    def getNumRoutes(self):
        return len(self._route)
    
    #def getMeters(self):
    #    return self._meters
    
class ioInputJSON(ioInput):
    def __init__(self, filename):
        super().__init__()

        with open(filename) as json_file:
            json_data = json.load(json_file)

        self._route.append(json_data["stop_list"])
        #self._meters = json_data["meters"]

    
