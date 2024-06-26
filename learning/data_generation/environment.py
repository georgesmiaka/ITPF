import time
import subprocess
from enum import Enum
import numpy as np

class Weather(Enum):
    ClearNoon = 0
    ClearSunset = 0
    CloudyNoon = 1
    CloudySunset = 1
    WetNoon = 2
    WetSunset = 2
    WetCloudySunset = 2
    WetCloudyNoon = 2
    SoftRainNoon = 2
    SoftRainSunset = 2
    MidRainyNoon = 3
    HardRainNoon = 3
    MidRainSunset = 3
    HardRainSunset = 3
    Default = 4

class Environment:
    def __init__(self):
        print("Initializing the environment...")

    def _open_server(self):
        cmd = ["./main.sh"]
        p = subprocess.Popen(cmd)
        time.sleep(10)
        return p

    def get_weather(self, weather):
        return Weather[weather].value

    def get_heading(self, temp_heading):
        return [x[1] for x in temp_heading]

    def get_positions(self, temp_pos):
        return [x[0] for x in temp_pos]

    def get_speed(self, temp_vel):
        vel = temp_vel[1]
        return 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def get_acceleration(self, acc):
        return np.sqrt(acc[1].x**2 + acc[1].y**2 + acc[1].z**2)
