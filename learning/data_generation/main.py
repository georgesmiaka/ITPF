from scenario import Scenario
from environment import Environment
from simulation import Simulation
from datetime import datetime

if __name__ == "__main__":
    # Example: Change these values to run different scenarios and maps
    
    map_file = "./scenes/Town05.xodr"
    map_name = "Town05"
    scene = "./scenes/activity_t05.scenic"
    max_trips = 1024
    save_data_dir = "output/train/activity_t05/"+f"trip"
    render = False

    # Create Scenario
    scenario = Scenario()
    scenario.fit_scenario(
        map_file=map_file,
        map_name=map_name,
        scene=scene
    )

    # Create Environment
    environment = Environment()
    environment._open_server()

    # Setup and Run Simulation
    simulation = Simulation(
        environment=environment, 
        scenario=scenario,
        map_file=map_file, 
        map_name=map_name,  
        render=render)
    simulation.setup_simulator()
    simulation.run(max_trips, save_data_dir)
