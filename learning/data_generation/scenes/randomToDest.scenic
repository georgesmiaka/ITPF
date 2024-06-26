param map = localPath('Town05.xodr')
param lgsvl_map = "Town05"
#ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset.
#If no weather set its uniform dist. over the above
#param weather = ""

model scenic.simulators.carla.model

home = (204, -11, 0.0)
work = (-88, 88, 0.0)
school = (92.1, -139.9, 0.0)
restaurant = (38, 118, 0.0)
kindergarten = (-271, -46, 0.0)

#end = new OrientedPoint on (-88, 88, 0.0)
end = new OrientedPoint on (restaurant)
startpoint = Uniform(*network.lanes)

behavior Drive():
  take SetDestinationAction(end)

start = new OrientedPoint on startpoint
ego = new Car at start, with behavior Drive(), with rolename 'hero'

terminate when (distance from ego.position to end) < 5
record (ego.heading) as heading
record (roadDirection) as roadDirection
record (ego.velocity) as velocity
