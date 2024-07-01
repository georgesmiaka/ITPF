param map = localPath('Town01.xodr')
param lgsvl_map = "Town01"

model scenic.simulators.carla.model

home = (204, -11, 0.0)
work = (-88, 88, 0.0)
school = (92.1, -139.9, 0.0)
restaurant = (38, 118, 0.0)
kindergarten = (-271, -46, 0.0)

destinations = [home, work, school, restaurant, kindergarten]
d = Uniform(*destinations)
start = new OrientedPoint on Uniform(*network.lanes)

end = new OrientedPoint on (d)
startpoint = Uniform(*network.lanes)

behavior Drive():
  take SetDestinationAction(end)

start = new OrientedPoint on startpoint
ego = new Car at start, with behavior Drive(), with rolename 'hero'

terminate when (distance from ego.position to end) < 5
record (ego.heading) as heading
record (roadDirection) as roadDirection
record (ego.velocity) as velocity
