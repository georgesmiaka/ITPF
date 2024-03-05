"""
Simulation Run Script

MAP: Location of running scene
MAP_NAME: The name of the map
SCENE: Scene to run
RENDER: 0 false, 1 true, determines if we display the scene in carla

Good to know:
    - The scene needs to record heading, roadDirection and velocity, see posToDest for inspiration
    - The ego car needs to be named "hero"

How to run:
    - Run using the following command "python3 sim.py", no window simulator will open if RENDER=0
    - Make sure that carla is running in the background, you can do a headless version with "./run.sh" if you have carla in the same parent directory. This makes the script run faster

Outputs:
    - Prints a json that you can >> in a json file

"""

import datetime
from enum import Enum
import json
import math
import os
import random
import subprocess
import time
import faker
import scenic
from trimesh import Scene
from scenic.core.simulators import TerminationType
from scenic.simulators.carla import CarlaSimulator
from scenic.simulators.carla.misc import get_speed


MAXSTEPS = 240
fake = faker.Faker()
RENDER    = False

"""
WEATHER
    0 - Default
    1 - ClearNoon
    2 - CloudyNoon
    3 - WetNoon
    4 - WetCloudyNoon
    5 - MidRainyNoon
    6 - HardRainNoon
    7 - SoftRainNoon
    8 - ClearSunset
    9 - CloudySunset
    10 - WetSunset
    11 - WetCloudySunset
    12 - MidRainSunset
    13 - HardRainSunset
    14 - SoftRainSunset

TIME
    21-06 = night     = 0
    07-09 = rush      = 1
    10-13 = lunch     = 2
    14-15 = afternoon = 3
    16-17 = rush      = 4
    18-20 = dinner    = 5
"""

class Weather(Enum):
    ClearNoon      = 0 #clear
    ClearSunset    = 0 #clear
    CloudyNoon     = 1 #cloudy
    CloudySunset   = 1 #cloudy
    WetNoon        = 2 #wet
    WetSunset      = 2 #wet
    WetCloudySunset = 2 #wet
    WetCloudyNoon  = 2 #wet
    SoftRainNoon   = 2 #wet
    SoftRainSunset = 2 #wet
    MidRainyNoon    = 3 #rain
    HardRainNoon   = 3 #rain
    MidRainSunset  = 3 #rain
    HardRainSunset = 3 #rain
    Default        = 4 #unknown

scenario_data = {
        "map": "Town05.xodr",
        "map_name": "Town05",
        "scene": "randomDriving.scenic",
        "driver": 0,
        "max_trips": 99,
        "save_data": "output/json"
}


def _open_server():
    cmd = ["./run.sh"]
    p = subprocess.Popen(cmd)
    time.sleep(10)
    return p

def get_weather(weather):
    return Weather[weather].value

def get_speed(temp_vel):
    vel = temp_vel[1]
    return (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

def get_heading(temp_heading):
    heading = map(lambda x: x[1], temp_heading)
    return list(heading)

def get_positions(temp_pos):
    pos = list(map(lambda x: x[0], temp_pos))
    return pos

#Starting carla server
carlaSim = _open_server()
simulator = CarlaSimulator("Town05", "Town05.xodr", render=RENDER, timeout=20)


MAP       = scenario_data['map']
MAP_NAME  = scenario_data['map_name']
SCENE     = scenario_data['scene']
DRIVER    = scenario_data['driver']
MAX_TRIPS = scenario_data['max_trips']
SAVE_DATA = scenario_data['save_data']

scenario = scenic.scenarioFromFile(SCENE, mode2D=True)
#simulator = CarlaSimulator(MAP_NAME, MAP, render=RENDER, timeout=30)

print("Scenario read and simulator started")

trips = 0
while trips <= MAX_TRIPS:
    try:
        path = SAVE_DATA+"_"+str(trips)+".json"
        if os.path.isfile(path):
            trips += 1
            continue
        print("Starting trip:",trips,"for scenario:",SCENE)

        scenes, _ = scenario.generate()
        weather = get_weather(scenes.params["weather"])
        #simulation = simulator.simulate(scenes, maxSteps=MAXSTEPS) # Max timesteps should be set from scenes
        simulation = simulator.simulate(scenes)

        if simulation:
                result = simulation.result
                position = get_positions(result.trajectory)
                heading = get_heading(result.records['heading'])
                velocity = list(map(get_speed, result.records['velocity']))

                data = list(zip(position, heading, velocity))
                x = []

                for (p, h, v) in data:
                    xx = {
                        "weather": weather,
                        "x": p.x,
                        "y": p.y,
                        "z": p.z,
                        "heading": h,
                        "velocity": v,
                    }
                    x.append(xx)
                print("Done trip:",str(trips)+",","Terminated because:",result.terminationType.name+",","ToT:",datetime.datetime.now())

                with open(path, "w+") as outfile:
                    json.dump(x, outfile, indent=4)
                    trips += 1
    except Exception as e:
        print("Error occured, trip is not saved to file, restarting")     
        print(e)