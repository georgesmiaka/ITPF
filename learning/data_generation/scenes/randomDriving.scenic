param map = localPath('Town05.xodr')
param lgsvl_map = "Town05"
model scenic.simulators.carla.model

times = [60, 120, 180]
start = new OrientedPoint on Uniform(*network.lanes)
t = Uniform(*times)

behavior Drive():
    do AutopilotBehavior()

ego = new Car at start, with behavior Drive(), with rolename 'hero'

terminate when simulation().currentTime >= t * 30
record (ego.heading) as heading
record (roadDirection) as roadDirection
record (ego.velocity) as velocity
record (ego._road) as road
record (ego._lane) as lane
