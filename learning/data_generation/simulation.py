import os
import json
import datetime
from scenic.simulators.carla import CarlaSimulator
import numpy as np
import scenic
import matplotlib.pyplot as plt
from scenic.core.simulators import TerminationType
from scenic.simulators.carla.misc import get_speed

class Simulation:
    def __init__(self, environment, scenario, map_file, map_name, render=False, max_steps=240):
        self.environment = environment
        self.scenario = scenario
        self.render = render
        self.max_steps = max_steps
        self.simulator = None
        self.map_name = map_name
        self.map_file = map_file

    def setup_simulator(self):
        self.simulator = CarlaSimulator(self.map_name, self.map_file, render=self.render, timeout=20)

    def run(self, max_trips, save_data_dir):
        scenario_init = scenic.scenarioFromFile(self.scenario.get_scenario_scene(), mode2D=True)
        print("Scenario pass...")
        print("Simulator started...")
        trips = 0

        while trips <= max_trips:
            try:
                path = f"{save_data_dir}_{trips}.json"
                if os.path.isfile(path):
                    trips += 1
                    continue
                print(f"Starting trip: {trips} for scenario: {self.scenario.get_scenario_map_file()}")

                scenes, _ = scenario_init.generate()
                weather = self.environment.get_weather(scenes.params["weather"])
                simulation = self.simulator.simulate(scenes, maxSteps=self.max_steps)

                if simulation:
                    result = simulation.result
                    position = self.environment.get_positions(result.trajectory)
                    heading = self.environment.get_heading(result.records['heading'])
                    velocity = list(map(self.environment.get_speed, result.records['velocity']))
                    #acc = list(map(self.environment.get_acceleration, result.records['acc']))

                    #data = list(zip(position, heading, velocity, acc))
                    data = list(zip(position, heading, velocity))
                    x = []

                    for p, h, v in data:
                        xx = {
                            "weather": weather,
                            "x": p.x,
                            "y": p.y,
                            "z": p.z,
                            "heading": h,
                            "velocity": v,
                            #"acc": a,
                        }
                        x.append(xx)
                    print(f"Done trip: {trips}, Terminated because: {result.terminationType.name}, ToT: {datetime.datetime.now()}")

                    with open(path, "w+") as outfile:
                        json.dump(x, outfile, indent=4)
                    trips += 1
            except Exception as e:
                print("Error occurred, trip is not saved to file, restarting")
                print(e)
