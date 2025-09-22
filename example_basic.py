from nexat_trace import CorridorStrategy, RoutePlanner, TrackSystem
from nexat_trace.util import plot_geometry as pg
from shapely import LineString, from_wkt

POLYGON_DATA = "POLYGON ((0 0, 650 0, 650 200, 450 200, 450 350, 500 370, 500 700, 50 700, 50 450, 50 100, 0 75, 0 0)," \
                "(225 225, 250 225, 250 250, 225 250, 225 225))"


outer_border = from_wkt(POLYGON_DATA)

reference_line = LineString(
    [
        outer_border.centroid,
        (outer_border.centroid.x + 1.0, outer_border.centroid.y)
    ]
)

ts = TrackSystem.from_border(
    outer_border,
    14.0,
    reference_line,
    [0.5, 1.0, 1.0, 0.5]
)

planner = RoutePlanner()
planner.route_params.debug_prints = True

# enable plotting of track graph
# close the plot window to resume the planning process
planner.route_params.debug_plot_track_graph = True

planner.route_params.corridor_strategy = CorridorStrategy.DRIVE_NONE
planner.route_params.round_trip_route = False

route = planner.plan_route_from_track_system(ts, -1)


path = route.get_linestring()

# finally, plot the path
pg.plot_linestring_rainbow(path)
pg.show_plot()
