import os
import scenic

class Scenario:
    def __init__(self):
        self.map_file = None
        self.map_name = None
        self.scene = None
    
    def fit_scenario(self, map_file, map_name, scene):
        self.map_file = map_file
        self.map_name = map_name
        self.scene = scene

    def get_scenario_map_name(self):
        return self.map_name
    
    def get_scenario_map_file(self):
        return self.map_file
    
    def get_scenario_scene(self):
        return self.scene
