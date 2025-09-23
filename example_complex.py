import time
from typing import List

from nexat_trace import CorridorStrategy, RoutePlanner, TrackSystem
from nexat_trace.util import plot_geometry as pg
from shapely import LinearRing, LineString, from_wkt

wkt_data: str | None = None
with open("example_data/trace_field.txt", "r") as file:
    wkt_data = file.read()

outer_border = from_wkt(wkt_data)

reference_line = LineString(
    [
        outer_border.centroid,
        (outer_border.centroid.x, outer_border.centroid.y + 1.0)
    ]
)

ts = TrackSystem.from_border(
    outer_border,
    14.0,
    reference_line,
    [0.5, 1.0, 1.0, 0.5]
)


planner = RoutePlanner()

planner.route_params.weights.headland_cost_exponent = 1.5
planner.route_params.weights.headland_distance_factor = 1.0
planner.route_params.max_block_size = 0
planner.route_params.working_width = 14.0
planner.route_params.disable_pi_curves = True
planner.route_params.fully_circle_cutouts = False
planner.route_params.round_trip_route = False
planner.route_params.corridor_strategy = CorridorStrategy.DRIVE_NONE

obstacle_avoidance_segments: List[LinearRing] = []

# round 2m radius obstacle on some line
obstacle_avoidance_segments.append(
    ts.ab_lines.geoms[42].centroid.buffer(
        planner.route_params.working_width / 2 + 2
    ).exterior
)
# round 1m radius obstacle on some other line
obstacle_avoidance_segments.append(
    ts.ab_lines.geoms[123].centroid.buffer(
        planner.route_params.working_width / 2 + 1
    ).exterior
)

ts.obstacle_avoidance_segments = obstacle_avoidance_segments

thread = planner.plan_route_from_track_system_async(ts, 60 * 5)
while not planner.done():
    progress = planner.get_progress()
    print(f"Progress: {progress[0]}%  Stage: {progress[1]}")
    time.sleep(1.0)

thread.join()

route = planner.get_route()

# print sums of turn maneuvers in route
print(route.turns)

# print any warnings or messages that occurred while planning
# see planning_messages.py for more info
print("Route warnings:")
print(planner.pop_messages())

pg.plot_linestring_rainbow(route.get_linestring(), 1)
pg.show_plot()
