from scenario import Scenario
from environment import Environment
from simulation import Simulation
from datetime import datetime

if __name__ == "__main__":
    # Example: Change these values to run different scenarios and maps
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # Create the subdirectory with the current date and time
    map_file = "./scenes/Town05.xodr"
    map_name = "Town05"
    scene = "./scenes/randomDriving.scenic"
    max_trips = 2400
    save_data_dir = "output/train/"+f"ex_{current_time}"
    render = True

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
