import pandas as pd
import numpy as np 
import os 
velocity_models = [
            {
                'vp': [0.400, 0.700],
                'vs': [0.200, 0.350],
                'depth': [0.01, 0.03]
            },
            {
                'vp': [0.800, 1.200],
                'vs': [0.200, 0.600],
                'depth': [0.1, 0.3]
            },
            {
                'vp': [0.900, 1.600],
                'vs': [0.600, 1.300],
                'depth': [0.2, 0.5]
            },
            {
                'vp': [0.700, 0.900, 1.400],
                'vs': [0.300, 0.500, 0.950],
                'depth': [0.1, 0.4, 0.7]
            },
            {
                'vp': [1.100, 1.700, 2.200],
                'vs': [0.700, 1.300, 1.700],
                'depth': [0.05, 0.2, 0.4]
            },
            {
                'vp': [1.5, 2.0, 3.0],
                'vs': [0.5, 1.0, 2.0],
                'depth': [0.1, 0.3, 0.6]
            },
            {
                'vp': [2.0, 2.5, 3.5],
                'vs': [1.0, 1.5, 2.5],
                'depth': [0.05, 0.2, 0.4]
            },
            {
                'vp': [2.5, 3.0, 4.0],
                'vs': [0.8, 1.2, 2.2],
                'depth': [0.1, 0.3, 0.5]
            },
            {
                'vp': [1.8, 2.2, 3.5],
                'vs': [0.7, 1.0, 2.5],
                'depth': [0.05, 0.25, 0.45]
            },
            {
                'vp': [0.400, 0.800, 1.200],
                'vs': [0.200, 0.400, 0.600],
                'depth': [0.02, 0.06, 0.1]
            },
            {
                'vp': [0.700, 1.500, 2.000, 2.300],
                'vs': [0.500, 1.200, 1.700, 2.000],
                'depth': [0.1, 0.2, 0.3, 0.6]
            },
            {
                'vp': [0.600, 1.200, 1.800, 2.300, 2.800],
                'vs': [0.400, 0.900, 1.300, 1.800, 2.200],
                'depth': [0.2, 0.4, 0.6, 1.0, 1.4]
            }
        ]
source_configurations = [
    ["1.00 0.00 0.00 0.00", "0.00 1.00 0.00 0.00", "0.00 0.00 1.00 0.00"], # Mxx = 1, Myy = 1, Mzz = 1
    ["1.00 0.00 0.00 0.00", "0.00 0.00 0.00 0.00", "0.00 0.00 0.00 0.00"], # Mxx = 1
    ["0.00 0.00 0.00 0.00", "0.00 1.00 0.00 0.00", "0.00 0.00 0.00 0.00"], # Myy = 1
    ["0.00 0.00 0.00 0.00", "0.00 0.00 0.00 0.00", "0.00 0.00 1.00 0.00"], # Mzz = 1
    ["0.00 1.00 0.00 0.00", "1.00 0.00 0.00 0.00", "0.00 0.00 0.00 0.00"]  # Mxy = Myx = 1
]
input_files = ["red_s0.5_101rec.drs", "blue_s0.5_101rec.drs", "yellow_s0.5_101rec.drs", "cyan_s0.5_101rec.drs", "magenta_s0.5_101rec.drs"]
offsets = ["0.001 0.5108", "0.001 0.509", "0.001 0.5075", "0.001 0.5053", "0.001 0.5021"]

class Data:
    """For connecting and interacting with MongoDB."""

    def __init__(self):
        self.velocity_models=velocity_models
        self.source_configurations=source_configurations
        self.input_files=input_files
        self.offsets=offsets
        
    def read_cable_positions(self,file_path="cable_positions.in"):
        df=pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None, names=['Index', 'Latitude', 'Longitude'])

        return df 
    def read_distances(self,file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            distances = [float(line.split()[0]) for line in lines[1:]]
            return np.array(distances)
    def read_FR(self,file_path):
        return np.loadtxt(file_path)

#convert the lat and lon to meters 
    def latlon_to_meters(self,lat, lon, origin_lat, origin_lon):
        lat_m = (lat - origin_lat) * 111194.9
        lon_m = (lon - origin_lon) * (111194.9 * np.cos(np.radians(origin_lat)))
        return lat_m, lon_m
    
    def get_velocity_models(self):
        
        return self.velocity_models
    def get_source_configurations(self):
        return self.source_configurations

    def get_input_files(self):
        return self.input_files

    def get_offsets(self):
        return self.offsets