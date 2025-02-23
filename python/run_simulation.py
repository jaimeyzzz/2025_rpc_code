import argparse
import os
import time
import yaml
import spdlog
import numpy as np

from cloth_geometry import ClothGeometry
from cloth_solver import ClothSolver

# define some constants
RESOURCES_DIR = "../"
OUTPUT_DIR = "../output/"

# create timestamp as year-month-day-hora-minute-second string
def create_timestamp():
    t = time.localtime()
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", t)
    return timestamp

# parse command line arguments
parser = argparse.ArgumentParser(description='Run simulation.')
parser.add_argument('config', type=str, help='Path to the YAML configuration file.')
args = parser.parse_args()

# load YAML configuration file
with open(args.config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# parse configuration parameters
scene_name = config['scene_name']
solver_name = config['solver_name']

fps = config['fps']
frames = config['frames']
substeps = config['substeps']
geometry_file = config['geometry_file']

# create output directory
timestamp = create_timestamp()
sim_path = os.path.join(OUTPUT_DIR, scene_name, solver_name, timestamp)
os.makedirs(sim_path, exist_ok=True)

# Create a console sink
console_sink = spdlog.stdout_color_sink_mt()

# Create a file sink
file_sink = spdlog.basic_file_sink_mt(os.path.join(sim_path, "logs.txt"))

# Create a vector of sinks
sinks = [console_sink, file_sink]

# Create a multi sink logger

print(dir(spdlog))
logger = spdlog.SinkLogger("multi_sink_logger", sinks)

# log configuration
logger.info("Scene name: {}".format(config['scene_name']))
logger.info("Solver name: {}".format(config['solver_name']))
logger.info("FPS: {}".format(config['fps']))
logger.info("Frames: {}".format(config['frames']))
logger.info("Substeps: {}".format(config['substeps']))
logger.info("Geometry file: {}".format(config['geometry_file']))
logger.info("Output path: {}".format(config['output_path']))

# run simulation
logger.info("Running simulation...")

usd_file = os.path.join(RESOURCES_DIR, geometry_file)
geometry = ClothGeometry(usd_file)

frames_num = 60
FPS = 60
substeps_num = 100
dt = 1.0/FPS/substeps_num

solver = ClothSolver(geometry, FPS)
filepath = '{}/frame_{}.usd'.format(sim_path, 0)
solver.dump(filepath)

for frame_index in range(frames_num):
    for _ in range(substeps_num):
        solver.update(dt)

    filepath = '{}/frame_{}.usd'.format(sim_path, frame_index + 1)
    solver.dump(filepath)


